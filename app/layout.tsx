import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Gradient Atlas · 梯度图谱",
  description: "从仓库资源清单自动生成的机器学习学习导航。",
  openGraph: {
    title: "Gradient Atlas · 梯度图谱",
    description: "一份简单、持续维护的机器学习学习资源清单。",
    type: "website",
    images: [
      {
        url: "https://raw.githubusercontent.com/luca-888/ml-portal/main/public/og.png",
        width: 1672,
        height: 941,
        alt: "Gradient Atlas — A Personal Machine Learning Index",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: "Gradient Atlas · 梯度图谱",
    description: "一份简单、持续维护的机器学习学习资源清单。",
    images: ["https://raw.githubusercontent.com/luca-888/ml-portal/main/public/og.png"],
  },
  icons: {
    icon: "/favicon.svg",
    shortcut: "/favicon.svg",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="zh-CN">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
