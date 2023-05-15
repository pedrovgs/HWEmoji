import { Configuration } from "webpack";
import HtmlWebpackPlugin from "html-webpack-plugin";
import MiniCssExtractPlugin from "mini-css-extract-plugin";
import CssMinimizerPlugin from "css-minimizer-webpack-plugin";
import TerserPlugin from "terser-webpack-plugin";
import ImageMinimizerPlugin from "image-minimizer-webpack-plugin";

import { config, baseDirectory } from "./webpack.config.common";

const prod: Configuration = {
  ...config,
  devtool: "hidden-source-map",
  optimization: {
    minimize: true,
    minimizer: [
      new CssMinimizerPlugin({
        minimizerOptions: {
          preset: ["default", { discardComments: { removeAll: true } }],
        },
      }),
      new TerserPlugin({ terserOptions: { format: { comments: false } }, extractComments: false }),
      new ImageMinimizerPlugin({
        minimizer: {
          implementation: ImageMinimizerPlugin.imageminMinify,
          options: {
            plugins: [
              ["gifsicle", { interlaced: true }],
              ["jpegtran", { progressive: true }],
              ["optipng", { optimizationLevel: 5 }],
              [
                "svgo",
                {
                  plugins: [
                    {
                      name: "preset-default",
                      params: { overrides: { inlineStyles: { onlyMatchedOnce: false } }, removeDoctype: false },
                    },
                  ],
                },
              ],
            ],
          },
        },
      }),
    ],
  },
  plugins: (config.plugins ?? []).concat([
    new HtmlWebpackPlugin({
      title: "Home",
      hash: true,
      filename: "index.html",
      chunks: ["main"],
      template: "public/index.html",
      minify: true,
    }),
    new MiniCssExtractPlugin({
      filename: "[name].css",
      chunkFilename: "[id].css",
    }),
  ]),
};

export default prod;
