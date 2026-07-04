import type { PluginInput, PluginOptions, Hooks } from "@opencode-ai/plugin"

interface FallbackOptions {
  fallbacks: Record<string, string[]>
  maxRetries: number
  cooldownMs: number
}

function parse(opts: Record<string, unknown>): FallbackOptions {
  return {
    fallbacks: (opts.fallbacks as Record<string, string[]>) ?? {},
    maxRetries: (opts.maxRetries as number) ?? 3,
    cooldownMs: (opts.cooldownMs as number) ?? 300_000,
  }
}

function key(providerID: string, modelID: string): string {
  return `${providerID}/${modelID}`
}

const sessionModel = new Map<string, string>()
const sessionFailures = new Map<string, Set<string>>()
const sessionLastFailure = new Map<string, number>()

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
      if (event.type !== "session.error") return
      if (!switchModel) return

      const props = event.properties as Record<string, unknown>
      const sessionID = props.sessionID as string | undefined
      if (!sessionID) return

      const currentModel = sessionModel.get(sessionID)
      if (!currentModel) return

      const fallbackChain = opts.fallbacks[currentModel]
      if (!fallbackChain || fallbackChain.length === 0) return

      const now = Date.now()
      const lastFail = sessionLastFailure.get(sessionID) ?? 0
      if (now - lastFail < opts.cooldownMs) return

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

      try {
        await switchModel(sessionID, providerID, modelID)
        sessionLastFailure.set(sessionID, now)
        console.log(`[model-fallback] ${currentModel} -> ${nextModel} (session ${sessionID})`)
      } catch (e) {
        console.error(`[model-fallback] failed to switch session ${sessionID} to ${nextModel}:`, e)
      }
    },
  }
}


