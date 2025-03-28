import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';
import * as dotenv from 'dotenv';

dotenv.config();

const isGitLab = process.env.GITLAB_CI === 'true';
const url = isGitLab ? `https://mdk-fixstars-llm-9d6ba492389e69266893e0491396538d1ad631e8458eba.gitlab.io` : `https://fixstars.github.io`;
const baseUrl = isGitLab ? `/` : `/mdk/`;
const gitRepoUrl = isGitLab ? `https://gitlab.com/fixstars/llm/mdk` : `https://github.com/fixstars/mdk`;
const gitLabel = isGitLab ? 'GitLab' : 'GitHub';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

const config: Config = {
  title: 'MDK: Model Development Kit',
  tagline: 'MDK: Model Development Kit',
  favicon: 'img/favicon.ico',
  scripts: ['/js/custom.js'],

  // Set the production url of your site here
  url: url,
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: baseUrl,

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: 'Fixstars AI Booster Team',
  projectName: 'MDK: Model Development Kit',

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'ja',
    locales: ['ja'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          // Please change this to your repo.
          routeBasePath: "/"
        },
        blog: {
          showReadingTime: true,
          feedOptions: {
            type: ['rss', 'atom'],
            xslt: true,
          },
          onInlineTags: 'warn',
          onInlineAuthors: 'warn',
          onUntruncatedBlogPosts: 'warn',
        },
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    // Replace with your project's social card
    image: 'img/fixstars_log_print.png',
    docs: {
      sidebar: {
        hideable: false,
      },
    },
    navbar: {
      title: 'MDK: Model Development Kit',
      logo: {
        alt: 'Fixstars Logo',
        src: 'img/fixstars_logo_print.png',
      },
      items: [
        {
          to: '#',
          label: '↔',
          position: 'right',
          className: 'fill-container-button'
        },
        {
          href: gitRepoUrl,
          label: gitLabel,
          position: 'right',
        },

      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'More',
          items: [
            {
              label: gitLabel,
              href: gitRepoUrl,
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Fixstars Corporation.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['bash','yaml','toml','diff'],
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
