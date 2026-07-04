import type { Plugin } from "@opencode-ai/plugin/v2/promise"
import type { LanguageModelV3, LanguageModelV3StreamResult, LanguageModelV3CallOptions } from "@ai-sdk/provider"

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

function findNext(
  failed: ReadonlySet<string>,
  primary: string,
  fallbacks: Record<string, string[]>,
): string | null {
  if (!failed.has(primary)) return primary
  const chain = fallbacks[primary] ?? []
  for (const fb of chain) {
    if (!failed.has(fb)) return fb
  }
  return null
}

function wrapModel(
  primaryKey: string,
  original: LanguageModelV3,
  sdk: { languageModel(id: string): LanguageModelV3 },
  apiIdLookup: Record<string, string>,
  opts: FallbackOptions,
  state: { failed: Set<string>; retries: number; lastFailure: number; cooldownMs: number; maxRetries: number },
): LanguageModelV3 {
  const tryFallback = async <T>(
    call: (model: LanguageModelV3) => PromiseLike<T>,
  ): Promise<T> => {
    let lastErr: unknown
    let currentKey = primaryKey

    for (let attempt = 0; attempt <= state.maxRetries; attempt++) {
      if (state.failed.has(currentKey)) {
        state.retries++
        state.lastFailure = Date.now()

        const nextKey = findNext(state.failed, primaryKey, opts.fallbacks)
        if (!nextKey) throw lastErr ?? new Error(`[model-fallback] all models exhausted for ${primaryKey}`)

        const nextApiId = apiIdLookup[nextKey]
        if (!nextApiId) throw lastErr ?? new Error(`[model-fallback] unknown model: ${nextKey}`)

        const fbModel = sdk.languageModel(nextApiId)
        console.log(`[model-fallback] ${primaryKey} -> ${nextKey}`)
        currentKey = nextKey
        return await call(fbModel)
      }

      try {
        return await call(currentKey === primaryKey ? original : sdk.languageModel(apiIdLookup[currentKey]))
      } catch (err) {
        state.failed.add(currentKey)
        lastErr = err
        console.log(`[model-fallback] error on ${currentKey}:`, (err as Error)?.message ?? err)
      }
    }

    throw lastErr
  }

  return {
    specificationVersion: "v3" as const,
    provider: original.provider,
    modelId: original.modelId,
    supportedUrls: original.supportedUrls,

    doGenerate(options: LanguageModelV3CallOptions) {
      return tryFallback((m) => m.doGenerate(options))
    },

    doStream(options: LanguageModelV3CallOptions) {
      return tryFallback((m) => m.doStream(options))
    },
  }
}

export const plugin: Plugin = {
  id: "opencode-model-fallback",

  async setup(context) {
    const opts = parse(context.options)
    const fallbackKeys = Object.keys(opts.fallbacks)
    if (fallbackKeys.length === 0) return

    // Resolve opencode model IDs ("provider/id") to provider API IDs ("api-model-id-v3")
    const apiIdLookup: Record<string, string> = {}
    let catalogLoaded = false

    await context.catalog.transform((draft) => {
      for (const provider of draft.provider.list()) {
        for (const [modelID, info] of provider.models) {
          apiIdLookup[key(provider.provider.id, modelID)] = info.api.id
        }
      }
      catalogLoaded = true
    })

    // If catalog already loaded, transform runs synchronously
    if (!catalogLoaded) {
      console.warn("[model-fallback] catalog not yet loaded, fallback API ID resolution may be incomplete")
    }

    const state = {
      failed: new Set<string>(),
      retries: 0,
      lastFailure: 0,
      cooldownMs: opts.cooldownMs,
      maxRetries: opts.maxRetries,
    }

    const now = () => {
      if (state.lastFailure === 0) return false
      if (Date.now() - state.lastFailure > state.cooldownMs) {
        state.failed.clear()
        state.retries = 0
        return true
      }
      return false
    }

    context.aisdk.language((event) => {
      const modelKey = key(event.model.providerID, event.model.id)
      const hasEntry = fallbackKeys.includes(modelKey)
      const isFallbackOf = fallbackKeys.some((k) => (opts.fallbacks[k] ?? []).includes(modelKey))

      if (!hasEntry && !isFallbackOf) return
      if (!event.sdk?.languageModel) return

      now()

      const original = event.sdk.languageModel(event.model.api.id)
      if (!original) return

      event.language = wrapModel(modelKey, original, event.sdk, apiIdLookup, opts, state)
    })
  },
}

export default plugin
