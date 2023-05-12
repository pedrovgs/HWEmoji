import { Gemoji } from "gemoji";

export enum WorkMode {
  collectData,
  test,
}

export type ComponentsListeners = {
  onEmojiSelected: (gemoji: Gemoji) => void;
  onModeSelected: (mode: WorkMode) => void;
  onEmojiSaved: (points: Points) => void;
};

export interface AppState {
  selectedEmoji: Gemoji;
  selectedMode: WorkMode;
}

export interface Point {
  x: number;
  y: number;
}

export interface DataSample {
  emoji: string;
  emojiName: string;
  points: Points;
}

export type Points = Point[];

export interface Prediction {
  emojiLabel: string;
  probability: number;
}

export type HWEmojiPredictionResult = Prediction[];
