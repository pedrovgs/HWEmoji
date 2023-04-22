import { TextEncoder, TextDecoder } from "util";
global.TextEncoder = TextEncoder;
// @ts-ignore
global.TextDecoder = TextDecoder;

import { JSDOM } from "jsdom";
import { init } from "../app";

const { window } = new JSDOM("<!doctype html><html><body></body></html>");
// Save these two objects in the global space so that libraries/tests
// can hook into them, using the above doc definition.
global.document = window.document;
// global.window = window as typeof globalThis;
global.Image = window.Image;

describe("app", () => {
  beforeEach(() => {
    window.document.body.innerHTML = `
    <header class="heading"></header>
    <section class="section-1"></section>
    <script src="./bundle.js"></script>`;
  });
  it("should render without error", () => {
    init({
      className: ".card",
      title: "Beckhi",
      content: "./public/images/beckhi.jpg",
    });
  });
  it("should match snapshot", () => {
    init({
      className: ".card",
      title: "Beckhi",
      content: "./public/images/beckhi.jpg",
    });
    expect(window.document.body).toMatchSnapshot();
  });
});
