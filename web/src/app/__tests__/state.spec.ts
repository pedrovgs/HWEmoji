import { WorkMode } from "../domain/model";
import { defaultAppState, selectEmoji, selectMode } from "../domain/state";
import { gemoji } from "gemoji";

describe("State management", () => {
  it("Should use grinning emoji by default", () => {
    const expectedEmoji = gemoji[0];
    expect(defaultAppState.selectedEmoji).toEqual(expectedEmoji);
    expect(expectedEmoji.emoji).toEqual("😀");
  });

  it("Should use collect mode by default", () => {
    expect(defaultAppState.selectedMode).toEqual(WorkMode.collectData);
  });

  it("Should update state and use the new selected emoji", () => {
    const emojiToSelect = gemoji[1];
    const state = selectEmoji(defaultAppState, emojiToSelect);
    expect(state.selectedEmoji).toEqual(emojiToSelect);
  });

  it("Should update mode and use the new selected mode", () => {
    const modeToSelect = WorkMode.test;
    const state = selectMode(defaultAppState, modeToSelect);
    expect(state.selectedMode).toEqual(modeToSelect);
  });
});
