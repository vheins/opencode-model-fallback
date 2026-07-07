import type { PluginInput, PluginOptions, Hooks } from "@opencode-ai/plugin"

interface FallbackOptions {
  fallbacks: Record<string, string[]>
  maxRetries: number
  cooldownMs: number
}

function parse(opts: Record<string, unknown>): FallbackOptions {
  const raw = opts.fallbacks
  const fallbacks: Record<string, string[]> =
    raw && typeof raw === "object" && !Array.isArray(raw)
      ? (raw as Record<string, string[]>)
      : {}
  const maxRetries = typeof opts.maxRetries === "number" ? opts.maxRetries : 3
  const cooldownMs = typeof opts.cooldownMs === "number" ? opts.cooldownMs : 300_000
  return { fallbacks, maxRetries, cooldownMs }
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
 * Check if the error is likely retryable.
 *
 * - APIError: strict classification using statusCode/isRetryable
 * - UnknownError / unclassified: assume retryable (the error likely
 *   wraps a provider failure like rate-limit that switching models fixes)
 */
function isRetryableError(event: Record<string, unknown>): boolean {
  const props = event.properties as Record<string, unknown> | undefined
  if (props?.error) {
    const err = props.error as Record<string, unknown>
    const errName = err.name as string | undefined
    const data = err.data as Record<string, unknown> | undefined
    if (errName === "APIError") {
      if (data?.statusCode === 429 || data?.statusCode === 503) return true
      if (data?.isRetryable === true) return true
      return false
    }
    // UnknownError or any other — assume retryable since we can't
    // determine the nature of the provider error
    return true
  }
  return false
}

export default async function modelFallbackPlugin(
  input: PluginInput,
  options?: PluginOptions,
): Promise<Hooks> {
  const opts = parse(options ?? {})
  const fallbackKeys = Object.keys(opts.fallbacks)
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

      // Capture model from step-start events — this is the ONLY reliable way
      // to get model info for subagent sessions (chat.params doesn't fire for them).
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

      const currentModel = sessionModel.get(sessionID)
      if (!currentModel) return

      const fallbackChain = opts.fallbacks[currentModel]
      if (!fallbackChain || fallbackChain.length === 0) return

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
      failures.add(currentModel)

      const nextModel = fallbackChain.find((m) => !failures!.has(m))
      if (!nextModel) return

      const [providerID, modelID] = nextModel.includes("/")
        ? [nextModel.split("/")[0], nextModel.split("/")[1]!]
        : ["", nextModel]

      if (!providerID) {
        console.warn(`[model-fallback] no provider in fallback target "${nextModel}", using empty providerID`)
      }

      try {
        await switchModel(sessionID, providerID, modelID)
        sessionLastFailure.set(sessionID, now)
        console.log(`[model-fallback] ${currentModel} -> ${nextModel} (session ${sessionID}, retry ${retries})`)
      } catch (e) {
        failures.add(nextModel)
        console.error(`[model-fallback] failed to switch session ${sessionID} to ${nextModel}:`, e)
      }
    },
  }
}
