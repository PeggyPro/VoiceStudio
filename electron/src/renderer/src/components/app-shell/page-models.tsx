import { useQuery } from '@tanstack/react-query';
import { Link } from '@tanstack/react-router';
import { useTranslation } from 'react-i18next';
import { isBackendReachable } from '@shared/utils/backendStage';
import { familyIcons, type ModelFamily } from '@/features/settings/model-family';
import { useModelCatalogue } from '@/features/settings/model-catalogue-query';
import { useTranslationEngines } from '@/features/settings/translation-settings';
import { useBackendStatus } from '@/hooks/use-backend-status';
import { useDictationSelection } from '@/hooks/use-dictation-selection';
import { useEngines, engineFamilyState } from '@/hooks/use-engines';
import { apiJson } from '@/lib/api/client';
import { cn } from '@/lib/utils';
import {
  compactPageModelName,
  enginePageModel,
  llmPageModel,
  type LoadedPageModel,
  type PageLlmProviders,
  type PageLlmSkill,
  type PageModelIdentity,
} from './page-model-info';

export type PageModelSpec =
  | ModelFamily
  | {
      family: ModelFamily;
      skill?: string;
      engineId?: string;
      routing?: 'capture';
      identity?: PageModelIdentity;
    };
type ModelSpec = Exclude<PageModelSpec, string>;
type ModelProps = ModelSpec & { loaded: LoadedPageModel[] };

/** The shell owns live polling; these observers share its cache and invalidations. */
export function PageModels({ models }: { models: readonly PageModelSpec[] }) {
  const { t } = useTranslation();
  const backend = useBackendStatus();
  const inventory = useQuery({
    queryKey: ['loaded-models'],
    queryFn: () => apiJson<{ models: LoadedPageModel[] }>('/model/loaded'),
    enabled: models.length > 0 && isBackendReachable(backend.stage),
    staleTime: 5_000,
  });
  if (!models.length) return null;
  const unique = new Map(
    models.map((model) => {
      const spec = typeof model === 'string' ? { family: model } : model;
      return [
        [
          spec.family,
          spec.skill,
          spec.engineId,
          spec.routing,
          spec.identity?.engine,
          spec.identity?.model,
        ].join(':'),
        spec,
      ] as const;
    }),
  );
  const loaded = inventory.isError ? [] : (inventory.data?.models ?? []);
  return (
    <div
      role="region"
      aria-label={t('pageModels.title')}
      className="flex shrink-0 flex-wrap justify-end gap-2 border-b border-border/40 bg-muted/10 px-5 py-2"
    >
      {[...unique].map(([key, spec]) => (
        <PageFamilyModel key={key} {...spec} loaded={loaded} />
      ))}
    </div>
  );
}

function PageFamilyModel(props: ModelProps) {
  if (props.identity)
    return (
      <ModelCard family={props.family} identity={props.identity} pending={false} error={false} />
    );
  switch (props.family) {
    case 'dictation':
      return <DictationModel {...props} />;
    case 'diarisation':
      return <DiarisationModel {...props} />;
    case 'translation':
      return <TranslationModel {...props} />;
    case 'llm':
      return <LlmModel {...props} />;
    default:
      return <EngineModel {...props} family={props.family} />;
  }
}

function EngineModel({ family, engineId, loaded }: ModelProps & { family: 'tts' | 'asr' }) {
  const engines = useEngines();
  const catalogue = useModelCatalogue();
  const selected = engineFamilyState(engines.data, family);
  const selection =
    engineId && selected && engineId !== selected.active
      ? { ...selected, active: engineId, active_model: null }
      : selected;
  return (
    <ModelCard
      family={family}
      identity={enginePageModel(family, selection, catalogue.data?.models, loaded)}
      pending={engines.isLoading}
      error={engines.isError}
    />
  );
}

function DictationModel({ family, routing }: ModelProps) {
  const query = useDictationSelection();
  const selected = query.data;
  // Live transcription pins the saved Sherpa model; calls use automatic capture routing.
  const capture = routing === 'capture';
  const modelId = capture ? selected?.capture?.model_id : selected?.model_id;
  return (
    <ModelCard
      family={family}
      identity={{
        model: capture
          ? selected?.capture?.repo_id || modelId || undefined
          : selected?.model?.repo_id || modelId || undefined,
        label: capture ? selected?.capture?.label : selected?.model?.label,
        // Capture inventory also includes lazy wrappers with no recognizer loaded.
        loaded: false,
        available: capture
          ? Boolean(selected?.capture?.available)
          : Boolean(selected?.available && selected.model?.installed),
      }}
      pending={query.isPending}
      error={query.isError}
    />
  );
}

interface DiarisationStatus {
  active: string;
  label?: string;
  model?: string;
  installed: boolean;
  loaded: boolean;
}

function DiarisationModel({ family }: ModelProps) {
  const backend = useBackendStatus();
  const query = useQuery({
    queryKey: ['diarisation-status'],
    queryFn: () => apiJson<DiarisationStatus>('/engines/diarisation'),
    enabled: isBackendReachable(backend.stage),
    staleTime: 30_000,
  });
  return (
    <ModelCard
      family={family}
      identity={{
        model: query.data?.model,
        label: query.data?.label,
        loaded: Boolean(query.data?.loaded),
        available: Boolean(query.data?.installed),
        inactive: query.data?.active === 'off',
      }}
      pending={query.isPending}
      error={query.isError}
    />
  );
}

function TranslationModel(props: ModelProps) {
  const query = useTranslationEngines();
  const active = props.engineId || query.data?.active;
  const engine = query.data?.engines.find((entry) => entry.id === active);
  if (engine?.category === 'llm')
    return <LlmModel {...props} skill={props.skill || 'dub_translation'} />;
  const resident = props.loaded.find((model) => model.id === 'translation:' + active);
  return (
    <ModelCard
      family={props.family}
      identity={{
        model: active === 'nllb' ? 'facebook/nllb-200-distilled-600M' : resident?.checkpoint,
        label: engine?.display_name,
        engine: engine?.display_name,
        loaded: Boolean(resident),
        available: Boolean(engine && (engine.ready ?? engine.installed)),
        inactive: active === 'off',
      }}
      pending={query.isPending}
      error={query.isError}
    />
  );
}

function LlmModel({ family, skill }: ModelProps) {
  const { t } = useTranslation();
  const backend = useBackendStatus();
  const enabled = isBackendReachable(backend.stage);
  const providers = useQuery({
    queryKey: ['llm-providers'],
    queryFn: ({ signal }) => apiJson<PageLlmProviders>('/api/settings/llm-providers', { signal }),
    enabled,
    staleTime: 30_000,
  });
  const skills = useQuery({
    queryKey: ['llm-skills'],
    queryFn: ({ signal }) =>
      apiJson<{ skills: PageLlmSkill[] }>('/api/settings/llm-skills', { signal }),
    enabled: enabled && Boolean(skill),
    staleTime: 30_000,
  });
  const selectedSkill = skills.data?.skills.find((entry) => entry.id === skill);
  return (
    <ModelCard
      family={family}
      identity={
        skill && !selectedSkill
          ? { loaded: false, available: false }
          : llmPageModel(providers.data, selectedSkill)
      }
      context={selectedSkill ? t(selectedSkill.name_key) : undefined}
      pending={providers.isPending || (Boolean(skill) && skills.isPending)}
      error={providers.isError || (Boolean(skill) && skills.isError)}
    />
  );
}

function ModelCard({
  family,
  identity,
  context,
  pending,
  error,
}: {
  family: ModelFamily;
  identity: PageModelIdentity;
  context?: string;
  pending: boolean;
  error: boolean;
}) {
  const { t } = useTranslation();
  const backend = useBackendStatus();
  const reachable = isBackendReachable(backend.stage);
  const state = !reachable
    ? 'offline'
    : pending
      ? 'loading'
      : error || (!identity.inactive && !identity.available)
        ? 'unavailable'
        : identity.inactive
          ? 'inactive'
          : identity.loaded
            ? 'loaded'
            : 'selected';
  const stateLabels = {
    offline: 'modelMaintenance.offline',
    loading: 'common.loading',
    unavailable: 'modelSettings.unavailable',
    inactive: 'engineSidebar.inactive',
    loaded: 'modelMaintenance.inMemory',
    selected: 'modelSettings.selected',
  } as const;
  const familyLabel = t('engineSidebar.' + family);
  const name = compactPageModelName(identity) || t('pageModels.unknownModel');
  const status = t(stateLabels[state]);
  const open = t('pageModels.openSettings', { family: familyLabel });
  const detail = [
    familyLabel + ' / ' + name,
    identity.model,
    identity.engine,
    context,
    status,
    open,
  ]
    .filter(Boolean)
    .join('\n');
  const Icon = familyIcons[family];
  return (
    <Link
      to="/settings/models/$family"
      params={{ family }}
      title={detail}
      aria-label={[familyLabel + ' / ' + name, context, status, open].filter(Boolean).join(' · ')}
      data-model-family={family}
      data-model-state={state}
      className="flex min-w-0 max-w-full items-center gap-2 rounded-lg border border-border/70 bg-background/80 px-2.5 py-1.5 text-xs shadow-sm transition-colors hover:border-primary/40 hover:bg-muted/60 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring sm:max-w-sm"
    >
      <Icon className="size-3.5 shrink-0 text-muted-foreground" aria-hidden="true" />
      <span className="shrink-0 font-medium text-muted-foreground">{familyLabel}</span>
      <span className="text-border" aria-hidden="true">
        /
      </span>
      <span className="min-w-0 truncate font-medium">{name}</span>
      {context && (
        <span className="max-w-24 truncate text-[10px] text-muted-foreground">{context}</span>
      )}
      <span
        aria-hidden="true"
        className={cn(
          'ml-1 size-1.5 shrink-0 rounded-full bg-muted-foreground/50',
          state === 'loaded' && 'bg-emerald-500',
          state === 'selected' && 'bg-primary/70',
          state === 'unavailable' && 'bg-amber-500',
        )}
      />
      <span className="shrink-0 text-[10px] text-muted-foreground">{status}</span>
    </Link>
  );
}
