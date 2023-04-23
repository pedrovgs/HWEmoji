import { Gemoji } from "gemoji";

export enum WorkMode {
  collectData,
  test,
}

export type ComponentsListeners = {
  onEmojiSelected: (gemoji: Gemoji) => void;
  onModeSelected: (mode: WorkMode) => void;
};

export interface AppState {
  selectedEmoji: Gemoji;
  selectedMode: WorkMode;
}

export interface Point {
  x: number;
  y: number;
}
