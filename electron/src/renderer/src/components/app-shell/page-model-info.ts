import type { CatalogueModel } from '@/features/settings/model-catalogue-query';
import type { EngineFamilyState } from '@/lib/api/types';

export interface LoadedPageModel {
  id: string;
  checkpoint: string;
  device?: string;
  engine_id?: string;
  is_active_engine?: boolean | null;
}

export interface PageModelIdentity {
  model?: string;
  label?: string;
  engine?: string;
  loaded: boolean;
  available: boolean;
  inactive?: boolean;
}

/** Selection wins over resident weights left behind by a previous engine/model. */
export function enginePageModel(
  family: 'tts' | 'asr' | 'llm',
  selection: EngineFamilyState | undefined,
  catalogue: CatalogueModel[] = [],
  loaded: LoadedPageModel[] = [],
): PageModelIdentity {
  if (!selection?.active || selection.active === 'off')
    return { loaded: false, available: false, inactive: true };
  const engine = selection.backends.find((entry) => entry.id === selection.active);
  const candidates = catalogue.filter(
    (entry) =>
      entry.role.toLowerCase() === family &&
      !entry.dictation_id &&
      Boolean(selection?.active && entry.engines?.includes(selection.active)),
  );
  const residents = loaded.filter(
    (entry) =>
      family === 'tts' &&
      (entry.engine_id === selection.active ||
        (entry.id === 'tts' && selection.active === 'omnivoice')),
  );
  const residentModel = residents.find((entry) => entry.checkpoint !== selection.active);
  // An installed model is not evidence of selection when a backend has variants.
  const model =
    selection.active_model ||
    (candidates.length === 1 ? candidates[0].repo_id : undefined) ||
    residentModel?.checkpoint;
  const matchingResident =
    family === 'tts'
      ? residents.some((entry) => entry.checkpoint === model) ||
        (!residentModel &&
          candidates.length === 1 &&
          candidates[0].repo_id === model &&
          residents.length > 0)
      : family === 'asr' && loaded.some((entry) => entry.id === 'asr' && entry.checkpoint === model);
  return {
    model,
    label: catalogue.find((entry) => entry.repo_id === model)?.label,
    engine: engine?.display_name || selection?.active || undefined,
    loaded: matchingResident,
    available: Boolean(engine?.available),
    inactive: !selection?.active || selection.active === 'off',
  };
}

export interface PageLlmProviders {
  active: string | null;
  engine_active: string;
  providers: {
    id: string;
    display_name: string;
    model: string;
    configured: boolean;
  }[];
}

export interface PageLlmSkill {
  id: string;
  name_key: string;
  enabled: boolean;
  ready: boolean;
  provider_override: string | null;
}

export function llmPageModel(
  providers: PageLlmProviders | undefined,
  skill?: PageLlmSkill,
): PageModelIdentity {
  const providerId = skill?.provider_override || providers?.active;
  const provider = providers?.providers.find((entry) => entry.id === providerId);
  return {
    model: provider?.model || undefined,
    engine: provider?.display_name,
    // Provider APIs report configuration, not whether a remote model is resident.
    loaded: false,
    available: skill
      ? skill.ready
      : providers?.engine_active === 'openai-compat' && Boolean(provider?.configured),
    inactive: skill
      ? !skill.enabled
      : !providers?.engine_active || providers.engine_active === 'off',
  };
}

export function compactPageModelName(identity: PageModelIdentity): string | undefined {
  const label = identity.label?.replace(/\s+\([^)]*\)\s*$/, '').trim();
  if (label) return label;
  return identity.model?.split(/[\\/]/).filter(Boolean).pop();
}
