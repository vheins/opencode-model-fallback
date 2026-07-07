import type { PluginInput, PluginOptions, Hooks } from "@opencode-ai/plugin"

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

/**
 * Extract sessionID from an event, handling both v1 (properties.sessionID)
 * and v2 (data.sessionID) event shapes.
 */
function extractSessionID(event: Record<string, unknown>): string | undefined {
  const props = event.properties as Record<string, unknown> | undefined
  if (props?.sessionID) return props.sessionID as string
  const data = event.data as Record<string, unknown> | undefined
  if (data?.sessionID) return data.sessionID as string
  return undefined
}

/**
 * Extract the status object from a session.status event.
 */
function extractStatus(event: Record<string, unknown>): { type?: string; message?: string } | undefined {
  const props = event.properties as Record<string, unknown> | undefined
  if (props?.status) return props.status as { type?: string; message?: string }
  const data = event.data as Record<string, unknown> | undefined
  if (data?.status) return data.status as { type?: string; message?: string }
  return undefined
}

/**
 * Extract the error message from an event for debugging.
 * Handles v1/v2 ApiError and SessionErrorUnknown shapes.
 */
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

/**
 * Returns true if the error should trigger a model fallback.
 *
 * For APIError: strict check for 429/503/isRetryable; everything else is
 * assumed non-retryable (prevents wasting fallback on bad requests etc.).
 * For all other error types (UnknownError etc.): assume retryable, because
 * the error likely wraps a transient provider failure like rate limiting.
 * The maxRetries + cooldownMs mechanism prevents runaway loops.
 */
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
    // UnknownError — assume retryable (likely wraps a provider failure)
    return true
  }
  return false
}

/**
 * Core fallback logic: find the next untried model in the fallback chain
 * and call switchModel. Shared between error events and status-retry events.
 */
async function performFallback(
  sessionID: string,
  switchModel: ((sessionID: string, providerID: string, modelID: string) => Promise<void>) | undefined,
  opts: FallbackOptions,
): Promise<void> {
  if (!switchModel) {
    console.log(`[model-fallback] session ${sessionID}: switchModel not available, skipping`)
    return
  }

  const currentModel = sessionModel.get(sessionID)
  
  // Try model-specific chain first, fall back to defaultFallbacks
  const fallbackChain = currentModel
    ? (opts.fallbacks[currentModel] ?? opts.defaultFallbacks)
    : opts.defaultFallbacks

  if (!fallbackChain || fallbackChain.length === 0) {
    console.log(`[model-fallback] session ${sessionID}: no fallback chain${currentModel ? ` for ${currentModel}` : ""}, skipping`)
    return
  }

  const now = Date.now()
  const lastFail = sessionLastFailure.get(sessionID) ?? 0

  // Cooldown expired → reset state so primary model gets retried
  if (now - lastFail >= opts.cooldownMs) {
    sessionFailures.delete(sessionID)
    sessionRetries.delete(sessionID)
  }

  // Enforce maxRetries
  const retries = (sessionRetries.get(sessionID) ?? 0) + 1
  sessionRetries.set(sessionID, retries)
  if (retries > opts.maxRetries) {
    console.log(`[model-fallback] session ${sessionID}: max retries (${opts.maxRetries}) reached`)
    return
  }

  // Find next untried model in the fallback chain
  let failures = sessionFailures.get(sessionID)
  if (!failures) {
    failures = new Set()
    sessionFailures.set(sessionID, failures)
  }
  if (currentModel) failures.add(currentModel)

  const nextModel = fallbackChain.find((m) => !failures!.has(m))
  if (!nextModel) {
    console.log(`[model-fallback] session ${sessionID}: all fallback models exhausted for ${currentModel}`)
    return
  }

  const [providerID, modelID] = nextModel.includes("/")
    ? [nextModel.split("/")[0], nextModel.split("/")[1]!]
    : ["", nextModel]

  if (!providerID) {
    console.warn(`[model-fallback] no provider in fallback target "${nextModel}", using empty providerID`)
  }

  try {
    console.log(`[model-fallback] switching session ${sessionID}: ${currentModel} -> ${nextModel} (attempt ${retries})`)
    await switchModel(sessionID, providerID, modelID)
    sessionLastFailure.set(sessionID, now)
    console.log(`[model-fallback] switch successful: ${currentModel} -> ${nextModel} (session ${sessionID})`)
  } catch (e) {
    failures.add(nextModel)
    console.error(`[model-fallback] switch failed: session ${sessionID} to ${nextModel}:`, e)
  }
}

export default async function modelFallbackPlugin(
  input: PluginInput,
  options?: PluginOptions,
): Promise<Hooks> {
  console.log("[model-fallback] plugin init called")
  const opts = parse(options ?? {})
  const fallbackKeys = Object.keys(opts.fallbacks)
  console.log(`[model-fallback] loaded, fallbacks: ${fallbackKeys.length ? fallbackKeys.join(", ") : "none"}${opts.defaultFallbacks ? `, default: ${opts.defaultFallbacks.join(", ")}` : ""}, maxRetries: ${opts.maxRetries}, cooldownMs: ${opts.cooldownMs}`)
  if (fallbackKeys.length === 0) return {}

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
    console.warn("[model-fallback] couldn't initialize v2 client; model switching disabled")
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
      console.log(`[model-fallback] event: ${eventType}`)

      // Capture model from session.created events (fires for ALL sessions
      // including subagents, before any step starts).
      if (eventType === "session.created") {
        let sessionID: string | undefined
        let model: { id?: string; providerID?: string } | undefined

        // v1: properties.info.model
        const props = eventAny.properties as Record<string, unknown> | undefined
        if (props?.sessionID && props?.info) {
          const info = props.info as Record<string, unknown>
          sessionID = props.sessionID as string
          model = info.model as { id?: string; providerID?: string } | undefined
        }
        // v2: data.info.model
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
          console.log(`[model-fallback] captured model for session ${sessionID}: ${k}`)
        }
        return
      }

      // Capture model from step-start events (backup for sessions that
      // already started before the plugin was loaded).
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

      // Handle session.status with retry type (server scheduled a retry, e.g.
      // "Free usage exceeded" goes to retry state instead of emitting an error)
      if (eventType === "session.status") {
        const status = extractStatus(eventAny)
        console.log(`[model-fallback] session.status: ${status?.type}`)
        if (status?.type === "retry" && status.message) {
          const sid = extractSessionID(eventAny)
          if (sid) {
            const safeMsg = status.message.length > 200 ? status.message.slice(0, 200) + "..." : status.message
            console.log(`[model-fallback] session ${sid} retry: ${safeMsg}`)
            await performFallback(sid, switchModel, opts)
          }
        }
        return
      }

      // Handle session-level errors and step-level failures (e.g. stream rate-limit)
      if (eventType !== "session.error" && eventType !== "session.next.step.failed") return
      if (!switchModel) return

      const sessionID = extractSessionID(eventAny)
      if (!sessionID) return

      // Log the actual error to help users debug why fallback triggered
      const errMsg = extractErrorMessage(eventAny)
      if (errMsg) {
        const safeMsg = errMsg.length > 200 ? errMsg.slice(0, 200) + "..." : errMsg
        console.log(`[model-fallback] session ${sessionID} error: ${safeMsg}`)
      }

      // Only trigger on retryable errors (rate limit, server error, etc.)
      if (!isRetryableError(eventAny)) return

      await performFallback(sessionID, switchModel, opts)
    },
  }
}
