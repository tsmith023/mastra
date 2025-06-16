import { ActionBarPrimitive, MessagePrimitive, ToolCallContentPartComponent, useMessage } from '@assistant-ui/react';
import { CheckIcon, CopyIcon } from 'lucide-react';
import { FC } from 'react';

import { MarkdownText } from './markdown-text';
import { TooltipIconButton } from './tooltip-icon-button';
import { ToolFallback } from '@/components/assistant-ui/tool-fallback';

export const AssistantMessage: FC<{ ToolFallback?: ToolCallContentPartComponent }> = ({
  ToolFallback: ToolFallbackCustom,
}) => {
  const data = useMessage();
  const isSolelyToolCall = data.content.length === 1 && data.content[0].type === 'tool-call';

  return (
    <MessagePrimitive.Root className="max-w-full">
      <div className="text-icon6 text-ui-lg leading-ui-lg">
        <MessagePrimitive.Content
          components={{
            Text: MarkdownText,
            tools: { Fallback: ToolFallbackCustom || ToolFallback },
          }}
        />
      </div>

      <div className="h-6 pt-1">{!isSolelyToolCall && <AssistantActionBar />}</div>
    </MessagePrimitive.Root>
  );
};

const AssistantActionBar: FC = () => {
  return (
    <ActionBarPrimitive.Root
      hideWhenRunning
      autohide="always"
      autohideFloat="single-branch"
      className="flex gap-1 items-center transition-all relative"
    >
      {/* <MessagePrimitive.If speaking={false}>
        <ActionBarPrimitive.Speak asChild>
          <TooltipIconButton tooltip="Read aloud">
            <AudioLinesIcon />
          </TooltipIconButton>
        </ActionBarPrimitive.Speak>
      </MessagePrimitive.If>
      <MessagePrimitive.If speaking>
        <ActionBarPrimitive.StopSpeaking asChild>
          <TooltipIconButton tooltip="Stop">
            <StopCircleIcon />
          </TooltipIconButton>
        </ActionBarPrimitive.StopSpeaking>
      </MessagePrimitive.If> */}
      <ActionBarPrimitive.Copy asChild>
        <TooltipIconButton tooltip="Copy" className="bg-transparent text-icon3 hover:text-icon6">
          <MessagePrimitive.If copied>
            <CheckIcon />
          </MessagePrimitive.If>
          <MessagePrimitive.If copied={false}>
            <CopyIcon />
          </MessagePrimitive.If>
        </TooltipIconButton>
      </ActionBarPrimitive.Copy>
      {/* <ActionBarPrimitive.Reload asChild>
        <TooltipIconButton tooltip="Refresh">
          <RefreshCwIcon />
        </TooltipIconButton>
      </ActionBarPrimitive.Reload> */}
    </ActionBarPrimitive.Root>
  );
};
