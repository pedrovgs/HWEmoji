import { initUIComponents, updateEmojiPreview } from "./components/components";
import log from "./log/logger";
import { Gemoji } from "gemoji";
import { Points, WorkMode } from "./domain/model";
import { defaultAppState, selectEmoji, selectMode } from "./domain/state";
import { saveDataSample } from "./data/DataSaver";
import { predictEmoji } from "./ai/HWEmoji";

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
    onEmojiSaved: (points: Points) => {
      if (points.length == 0) {
        return;
      }
      predictEmoji(points);
      const emoji = appState.selectedEmoji;
      const sample = {
        emoji: emoji.emoji,
        emojiName: emoji.names[0],
        points: points,
      };
      //saveDataSample(sample);
    },
  };
  return listeners;
};
