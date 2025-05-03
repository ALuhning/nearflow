import resolve from '@rollup/plugin-node-resolve';
import commonjs from '@rollup/plugin-commonjs';
import typescript from '@rollup/plugin-typescript';
import external from 'rollup-plugin-peer-deps-external';

export default {
  input: 'src/index.tsx', // or your actual entry file
  output: {
    file: 'dist/bundle.js',
    format: 'esm',
    sourcemap: true,
  },
  external: [
    'react',
    'react-dom',
    'react/jsx-runtime',
    'react-router',
    'react-router-dom',
  ],
  plugins: [
    external(),
    resolve({
      extensions: ['.js', '.jsx', '.ts', '.tsx'],
    }),
    commonjs({
      include: /node_modules/,
      requireReturnsDefault: 'auto',
    }),
    typescript({
      tsconfig: './tsconfig.json',
    }),
  ],
};
