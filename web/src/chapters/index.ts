import { FunctionComponent } from 'preact'
import BigPicture from './BigPicture'
import Tokens from './Tokens'
import Embeddings from './Embeddings'
import Attention from './Attention'
import Block from './Block'
import NextToken from './NextToken'
import Backprop from './Backprop'
import Optimizer from './Optimizer'
import TrainingLoop from './TrainingLoop'
import Pretraining from './Pretraining'
import SFT from './SFT'
import Preferences from './Preferences'
import Pipeline from './Pipeline'
import AgentLoop from './AgentLoop'
import Harness from './Harness'
import ToolUse from './ToolUse'
import AgentTraining from './AgentTraining'
import Workflows from './Workflows'
import Tour from './Tour'
import Reasoning from './Reasoning'
import Evals from './Evals'
import Inference from './Inference'

export const chapterViews: Record<string, FunctionComponent> = {
  'tour': Tour,
  'big-picture': BigPicture,
  'tokens': Tokens,
  'embeddings': Embeddings,
  'attention': Attention,
  'block': Block,
  'next-token': NextToken,
  'backprop': Backprop,
  'optimizer': Optimizer,
  'training-loop': TrainingLoop,
  'pretraining': Pretraining,
  'sft': SFT,
  'preferences': Preferences,
  'reasoning': Reasoning,
  'evals': Evals,
  'pipeline': Pipeline,
  'inference': Inference,
  'agent-loop': AgentLoop,
  'harness': Harness,
  'tool-use': ToolUse,
  'agent-training': AgentTraining,
  'workflows': Workflows,
}
