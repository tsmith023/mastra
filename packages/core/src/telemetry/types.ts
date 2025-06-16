import type { SpanExporter, ReadableSpan } from '@opentelemetry/sdk-trace-base';

/** Sampling strategy configuration for OpenTelemetry */
export type SamplingStrategy =
  | {
      /** Sample traces based on a probability between 0 and 1 */
      type: 'ratio';
      /** Probability between 0 and 1 (e.g., 0.1 for 10% sampling) */
      probability: number;
    }
  | {
      /** Sample all traces */
      type: 'always_on';
    }
  | {
      /** Don't sample any traces */
      type: 'always_off';
    }
  | {
      /** Use parent sampling decision if available, otherwise use root sampler */
      type: 'parent_based';
      /** Configuration for the root sampler when no parent context exists */
      root: {
        /** Probability between 0 and 1 for the root sampler */
        probability: number;
      };
    };

/** Configuration options for OpenTelemetry */
export type OtelConfig = {
  /** Name of the service for telemetry identification */
  serviceName?: string;

  /** Whether telemetry is enabled. Defaults to true */
  enabled?: boolean;

  /** Name of the tracer to use. Defaults to 'mastra-tracer' */
  tracerName?: string;

  /** Sampling configuration to control trace data volume */
  sampling?: SamplingStrategy;

  /** Whether to disable local export */
  disableLocalExport?: boolean;

  /** Export configuration for sending telemetry data */
  export?:
    | {
        /** Export to an OTLP (OpenTelemetry Protocol) endpoint */
        type: 'otlp';
        /** Whether to use gRPC or HTTP for OTLP */
        protocol?: 'grpc' | 'http';
        /** OTLP endpoint URL */
        endpoint?: string;
        /** Optional headers for OTLP requests */
        headers?: Record<string, string>;
      }
    | {
        /** Export to console for development/debugging */
        type: 'console';
      }
    | {
        type: 'custom';
        tracerName?: string;
        exporter: SpanExporter;
      };
};

export type Trace = {
  id: string;
  parentSpanId: string;
  name: string;
  traceId: string;
  scope: string;
  kind: ReadableSpan['kind'];
  attributes: ReadableSpan['attributes'];
  status: ReadableSpan['status'];
  events: ReadableSpan['events'];
  links: ReadableSpan['links'];
  other: Record<string, any>;
  startTime: number;
  endTime: number;
  createdAt: string;
};
