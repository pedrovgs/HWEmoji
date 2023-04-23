import type { Config } from "@jest/types";

const config: Config.InitialOptions = {
  preset: "ts-jest",
  setupFiles: ["jest-canvas-mock"],
  moduleDirectories: ["node_modules", "src"],
  moduleFileExtensions: ["ts", "tsx", "js", "jsx", "json"],
  modulePathIgnorePatterns: ["dist", "build"],
  moduleNameMapper: {
    "src/(.*)": "<rootDir>/src/$1",
    "contexts(.*)$": "<rootDir>/src/contexts/$1",
    "\\.(css|less)$": "identity-obj-proxy",
  },
  transform: {
    "^.+\\.ts$": "ts-jest",
    "^.+\\.js$": "ts-jest",
    "^.+\\.tsx$": "ts-jest",
  },
  transformIgnorePatterns: ["node_modules/(?!gemoji)"],
  globalSetup: "./jest/global-setup.ts",
  verbose: true,
  testEnvironment: "jsdom",
};

export default config;
