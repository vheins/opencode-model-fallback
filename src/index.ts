import type { PluginInput, PluginOptions, Hooks } from "@opencode-ai/plugin"

console.log("[model-fallback] module loaded")

interface OpencodeClient {
  app: {
    log: (params: {
      body: {
        service: string
        level: "debug" | "info" | "warn" | "error"
        message: string
        extra?: Record<string, unknown>
      }
      query?: { directory?: string }
    }) => Promise<unknown>
  }
}

interface FallbackOptions {
  fallbacks: Record<string, string[]>
  defaultFallbacks?: string[]
  maxRetries: number
  cooldownMs: number
}

function parse(opts: Record<string, unknown>): FallbackOptions {
  const raw = opts.fallbacks
  const fallbacks: Record<string, string[]> =
    raw && typeof raw === "object" && !Array.isArray(raw)
      ? (raw as Record<string, string[]>)
      : {}
  const rawDefault = opts.defaultFallbacks
  const defaultFallbacks: string[] | undefined =
    Array.isArray(rawDefault) && rawDefault.length > 0
      ? (rawDefault as string[])
      : undefined
  const maxRetries = typeof opts.maxRetries === "number" ? opts.maxRetries : 3
  const cooldownMs = typeof opts.cooldownMs === "number" ? opts.cooldownMs : 300_000
  return { fallbacks, defaultFallbacks, maxRetries, cooldownMs }
}

function key(providerID: string, modelID: string): string {
  return `${providerID}/${modelID}`
}

const sessionModel = new Map<string, string>()
const sessionFailures = new Map<string, Set<string>>()
const sessionLastFailure = new Map<string, number>()
const sessionRetries = new Map<string, number>()

function extractSessionID(event: Record<string, unknown>): string | undefined {
  const props = event.properties as Record<string, unknown> | undefined
  if (props?.sessionID) return props.sessionID as string
  const data = event.data as Record<string, unknown> | undefined
  if (data?.sessionID) return data.sessionID as string
  return undefined
}

function extractStatus(event: Record<string, unknown>): { type?: string; message?: string } | undefined {
  const props = event.properties as Record<string, unknown> | undefined
  if (props?.status) return props.status as { type?: string; message?: string }
  const data = event.data as Record<string, unknown> | undefined
  if (data?.status) return data.status as { type?: string; message?: string }
  return undefined
}

function extractErrorMessage(event: Record<string, unknown>): string | undefined {
  const props = event.properties as Record<string, unknown> | undefined
  if (props?.error) {
    const err = props.error as Record<string, unknown>
    const data = err.data as Record<string, unknown> | undefined
    if (data?.message) return data.message as string
    if (err.message) return err.message as string
  }
  const data = event.data as Record<string, unknown> | undefined
  if (data?.error) {
    const err = data.error as Record<string, unknown>
    if (err.message) return err.message as string
    const errData = err.data as Record<string, unknown> | undefined
    if (errData?.message) return errData.message as string
  }
  return undefined
}

function isRetryableError(event: Record<string, unknown>): boolean {
  const props = event.properties as Record<string, unknown> | undefined
  if (props?.error) {
    const err = props.error as Record<string, unknown>
    const errName = err.name as string | undefined
    if (errName === "APIError") {
      const data = err.data as Record<string, unknown> | undefined
      if (data?.statusCode === 429 || data?.statusCode === 503) return true
      if (data?.isRetryable === true) return true
      return false
    }
    return true
  }
  return false
}

async function performFallback(
  sessionID: string,
  switchModel: ((sessionID: string, providerID: string, modelID: string) => Promise<void>) | undefined,
  opts: FallbackOptions,
  log: (level: string, msg: string) => void,
): Promise<void> {
  if (!switchModel) {
    log("info", `session ${sessionID}: switchModel not available, skipping`)
    return
  }

  const currentModel = sessionModel.get(sessionID)

  const fallbackChain = currentModel
    ? (opts.fallbacks[currentModel] ?? opts.defaultFallbacks)
    : opts.defaultFallbacks

  if (!fallbackChain || fallbackChain.length === 0) {
    log("info", `session ${sessionID}: no fallback chain${currentModel ? ` for ${currentModel}` : ""}, skipping`)
    return
  }

  const now = Date.now()
  const lastFail = sessionLastFailure.get(sessionID) ?? 0

  if (now - lastFail >= opts.cooldownMs) {
    sessionFailures.delete(sessionID)
    sessionRetries.delete(sessionID)
  }

  const retries = (sessionRetries.get(sessionID) ?? 0) + 1
  sessionRetries.set(sessionID, retries)
  if (retries > opts.maxRetries) {
    log("info", `session ${sessionID}: max retries (${opts.maxRetries}) reached`)
    return
  }

  let failures = sessionFailures.get(sessionID)
  if (!failures) {
    failures = new Set()
    sessionFailures.set(sessionID, failures)
  }
  failures.add(currentModel ?? `__unknown__`)

  const nextModel = fallbackChain.find((m) => !failures!.has(m))
  if (!nextModel) {
    log("info", `session ${sessionID}: all fallback models exhausted${currentModel ? ` for ${currentModel}` : ""}`)
    return
  }

  const [providerID, modelID] = nextModel.includes("/")
    ? [nextModel.split("/")[0], nextModel.split("/")[1]!]
    : ["", nextModel]

  if (!providerID) {
    log("warn", `no provider in fallback target "${nextModel}", using empty providerID`)
  }

  try {
    log("info", `switching session ${sessionID}: ${currentModel ?? "unknown"} -> ${nextModel} (attempt ${retries})`)
    await switchModel(sessionID, providerID, modelID)
    sessionLastFailure.set(sessionID, now)
    log("info", `switch successful: ${currentModel ?? "unknown"} -> ${nextModel} (session ${sessionID})`)
  } catch (e) {
    failures.add(nextModel)
    log("error", `switch failed: session ${sessionID} to ${nextModel}: ${e}`)
  }
}

export default async function modelFallbackPlugin(
  input: PluginInput,
  options?: PluginOptions,
): Promise<Hooks> {
  const client: any = (input as any).client
  const directory = input.directory

  // Log to BOTH stdout and structured logs for maximum visibility
  function log(level: string, msg: string): void {
    const prefix = `[model-fallback]`
    if (level === "error") {
      console.error(`${prefix} ${msg}`)
    } else if (level === "warn") {
      console.warn(`${prefix} ${msg}`)
    } else {
      console.log(`${prefix} ${msg}`)
    }
    try {
      if (client?.app?.log) {
        client.app.log({
          body: {
            service: "model-fallback",
            level: level as "debug" | "info" | "warn" | "error",
            message: msg,
            extra: { directory },
          },
          query: { directory },
        }).catch(() => {})
      }
    } catch {}
  }

  log("info", "plugin init called")

  const opts = parse(options ?? {})
  const fallbackKeys = Object.keys(opts.fallbacks)
  log("info", `loaded, fallbacks: ${fallbackKeys.length ? fallbackKeys.join(", ") : "none"}${opts.defaultFallbacks ? `, default: ${opts.defaultFallbacks.join(", ")}` : ""}, maxRetries: ${opts.maxRetries}, cooldownMs: ${opts.cooldownMs}`)
  if (fallbackKeys.length === 0 && !opts.defaultFallbacks) return {}

  let switchModel: ((sessionID: string, providerID: string, modelID: string) => Promise<void>) | undefined

  try {
    const mod = await import("@opencode-ai/sdk/v2/client")
    const v2Client = mod.createOpencodeClient({
      baseUrl: input.serverUrl.toString(),
      directory: input.directory,
    })
    switchModel = async (sessionID, providerID, modelID) => {
      await v2Client.v2.session.switchModel({
        sessionID,
        model: { id: modelID, providerID },
      })
    }
  } catch {
    log("warn", "couldn't initialize v2 client; model switching disabled")
  }

  return {
    "chat.params": async (chatInput, _output) => {
      const providerId = chatInput.provider?.info?.id
      const modelId = chatInput.model?.id
      if (providerId && modelId) {
        sessionModel.set(chatInput.sessionID, key(providerId, modelId))
      }
    },

    event: async ({ event }) => {
      const eventAny = event as Record<string, unknown>
      const eventType = eventAny.type as string
      log("info", `event: ${eventType}`)

      if (eventType === "session.created") {
        let sessionID: string | undefined
        let model: { id?: string; providerID?: string } | undefined

        const props = eventAny.properties as Record<string, unknown> | undefined
        if (props?.sessionID && props?.info) {
          const info = props.info as Record<string, unknown>
          sessionID = props.sessionID as string
          model = info.model as { id?: string; providerID?: string } | undefined
        }
        if (!sessionID) {
          const data = eventAny.data as Record<string, unknown> | undefined
          if (data?.sessionID && data?.info) {
            const info = data.info as Record<string, unknown>
            sessionID = data.sessionID as string
            model = info.model as { id?: string; providerID?: string } | undefined
          }
        }

        if (sessionID && model?.id && model?.providerID) {
          const k = key(model.providerID, model.id)
          sessionModel.set(sessionID, k)
          log("info", `captured model for session ${sessionID}: ${k}`)
        }
        return
      }

      if (eventType === "session.next.step.started") {
        const data = eventAny.data as Record<string, unknown> | undefined
        if (data?.sessionID && data?.model) {
          const model = data.model as { id?: string; providerID?: string }
          if (model.id && model.providerID) {
            sessionModel.set(data.sessionID as string, key(model.providerID, model.id))
          }
        }
        return
      }

      if (eventType === "session.status") {
        const status = extractStatus(eventAny)
        log("info", `session.status: ${status?.type}`)
        if (status?.type === "retry" && status.message) {
          const sid = extractSessionID(eventAny)
          if (sid) {
            const safeMsg = status.message.length > 200 ? status.message.slice(0, 200) + "..." : status.message
            log("info", `session ${sid} retry: ${safeMsg}`)
            await performFallback(sid, switchModel, opts, log)
          }
        }
        return
      }

      if (eventType !== "session.error" && eventType !== "session.next.step.failed") return
      if (!switchModel) return

      const sessionID = extractSessionID(eventAny)
      if (!sessionID) return

      const errMsg = extractErrorMessage(eventAny)
      if (errMsg) {
        const safeMsg = errMsg.length > 200 ? errMsg.slice(0, 200) + "..." : errMsg
        log("info", `session ${sessionID} error: ${safeMsg}`)
      }

      if (!isRetryableError(eventAny)) return

      await performFallback(sessionID, switchModel, opts, log)
    },
  }
}
