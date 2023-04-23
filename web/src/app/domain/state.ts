import { gemoji, Gemoji } from "gemoji";

import { AppState, WorkMode } from "./model";

export const defaultAppState: AppState = {
  selectedEmoji: gemoji[0],
  selectedMode: WorkMode.collectData,
};

export const selectEmoji = (state: AppState, emoji: Gemoji): AppState => {
  return {
    ...state,
    selectedEmoji: emoji,
  };
};

export const selectMode = (state: AppState, mode: WorkMode): AppState => {
  return {
    ...state,
    selectedMode: mode,
  };
};
