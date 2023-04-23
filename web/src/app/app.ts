import { initUIComponents, updateEmojiPreview } from "./components/components";
import log from "./log/logger";
import { Gemoji } from "gemoji";
import { WorkMode } from "./domain/model";
import { defaultAppState, selectEmoji, selectMode } from "./domain/state";

let appState = defaultAppState;

export const init = async () => {
  log("😃 Initializing HWEmoji ");
  const listeners = componentsListener();
  await initUIComponents(listeners, appState);
  log("💪 HWEmoji initialized");
};

const componentsListener = () => {
  const listeners = {
    onEmojiSelected: (gemoji: Gemoji) => {
      appState = selectEmoji(appState, gemoji);
      updateEmojiPreview(gemoji);
    },
    onModeSelected: (mode: WorkMode) => {
      appState = selectMode(appState, mode);
    },
  };
  return listeners;
};
