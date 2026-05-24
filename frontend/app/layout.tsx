import type { Metadata } from "next";
import "./globals.css";
import Analytics from "@/components/shell/Analytics";

export const metadata: Metadata = {
  title: "Aito Equity demo",
  description: "Replace with your demo's name & description.",
  icons: {
    icon: "/aito-favicon.svg",
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <Analytics />
        {children}
      </body>
    </html>
  );
}
