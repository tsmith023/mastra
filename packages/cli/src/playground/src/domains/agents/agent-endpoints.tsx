export function AgentEndpoints({ agentId }: { agentId: string }) {
  return (
    <div className="space-y-4 pt-2 px-4 pb-4 text-xs overflow-y-auto">
      <div className="grid grid-cols-[70px_1fr] gap-2">
        <p className="text-mastra-el-3">GET</p>
        <p className="text-mastra-el-5">/api/agents</p>
      </div>
      <div className="grid grid-cols-[70px_1fr] gap-2">
        <p className="text-mastra-el-3">GET</p>
        <p className="text-mastra-el-5">/api/agents/{agentId}</p>
      </div>
      <div className="grid grid-cols-[70px_1fr] gap-2">
        <p className="text-mastra-el-3">GET</p>
        <p className="text-mastra-el-5">/api/agents/{agentId}/evals/ci</p>
      </div>
      <div className="grid grid-cols-[70px_1fr] gap-2">
        <p className="text-mastra-el-3">GET</p>
        <p className="text-mastra-el-5">/api/agents/{agentId}/evals/live</p>
      </div>
      <div className="grid grid-cols-[70px_1fr] gap-2">
        <p className="text-mastra-el-3">POST</p>
        <p className="text-mastra-el-5">/api/agents/{agentId}/instructions</p>
      </div>
      <div className="grid grid-cols-[70px_1fr] gap-2">
        <p className="text-mastra-el-3">POST</p>
        <p className="text-mastra-el-5">/api/agents/{agentId}/generate</p>
      </div>
      <div className="grid grid-cols-[70px_1fr] gap-2">
        <p className="text-mastra-el-3">POST</p>
        <p className="text-mastra-el-5">/api/agents/{agentId}/stream</p>
      </div>
      <div className="grid grid-cols-[70px_1fr] gap-2">
        <p className="text-mastra-el-3">POST</p>
        <p className="text-mastra-el-5">/api/agents/{agentId}/text-object</p>
      </div>
      <div className="grid grid-cols-[70px_1fr] gap-2">
        <p className="text-mastra-el-3">POST</p>
        <p className="text-mastra-el-5">/api/agents/{agentId}/stream-object</p>
      </div>
    </div>
  );
}
