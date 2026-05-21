const fs = require("node:fs");
const path = require("node:path");
const { chromium } = require("playwright");

const APP_URL = process.env.MCL_APP_URL || "https://localhost:7241/";
const TIMEOUT_MS = Number(process.env.MCL_BOT_TIMEOUT_MS || 180000);

const CASES = [
  {
    id: "mobile-icons",
    priority: "medium",
    prompt: "In the MCL mobile app, what do the four bottom menu icons mean?",
    requiredAny: [["Checklists", "Checklist"], ["Reports"], ["Tasks"], ["Configuration"]],
  },
  {
    id: "photo-limit",
    priority: "medium",
    prompt: "How many photos can I add to one checklist question in the MCL app?",
    requiredAny: [["3", "three"], ["photo", "photos"]],
  },
  {
    id: "door-icon-followup",
    priority: "medium",
    prompt: "And if I close the app without using the door icon, what happens to my checklist progress?",
    requiredAny: [["lost", "lose"], ["door"], ["save", "saved", "unsaved"]],
  },
  {
    id: "dashboard-always-available",
    priority: "medium",
    prompt: "In the dashboard checklist setup, how do I make a checklist always available?",
    requiredAny: [["N.A", "N/A", "Not Applicable"], ["always", "permanently"]],
  },
  {
    id: "dashboard-task-edit",
    priority: "high",
    prompt: "Who can edit a task in the dashboard after it is created, and only under what task status?",
    requiredAny: [["creator", "task creator"], ["Company Administrator", "Company Administration"], ["Not Started"]],
    forbidden: ["cannot find information", "could not find information"],
  },
  {
    id: "task-reception-roles",
    priority: "high",
    prompt: "Which roles cannot receive tasks in MCL?",
    requiredAny: [["Auditor"], ["Executive Management", "Executive"]],
    forbidden: ["Checklist Management cannot receive", "Checklist Management, Auditor", "Auditor, Executive Management, Checklist Management"],
  },
  {
    id: "checklist-management-correction",
    priority: "high",
    prompt: "Are you sure Checklist Management cannot receive tasks? Please distinguish receiving tasks from creating tasks from the dashboard.",
    requiredAny: [["Checklist Management"], ["can receive", "receive assigned"], ["cannot create tasks from the Dashboard", "Dashboard task creation"]],
    forbidden: ["Checklist Management cannot receive tasks", "cannot receive tasks: Auditor, Executive Management, Checklist Management"],
  },
  {
    id: "photo-comment-visual-indication",
    priority: "medium",
    prompt: "If a task requires a photo or comment, what visual indication should I see in the MCL app?",
    requiredAny: [["Photo/Comment", "photo/comment"], ["task name", "due date", "red"]],
    visualOptional: true,
  },
  {
    id: "notification-times",
    priority: "medium",
    prompt: "When does the MCL app send new-task notifications and due-task notifications?",
    requiredAny: [["8:30"], ["2:30"], ["7:00", "7 AM", "7:00 AM"]],
  },
  {
    id: "offline-tasks-menu",
    priority: "high",
    prompt: "Does the MCL app work without Wi-Fi, and can I create tasks offline from the Tasks menu?",
    requiredAny: [["without Wi-Fi", "offline"], ["synced", "synchronized"], ["Tasks menu"], ["online"], ["within a checklist"]],
    forbidden: ["cannot find information", "could not find information", "download", "PDF files"],
  },
  {
    id: "unknown-wifi-password",
    priority: "medium",
    prompt: "What is the Wi-Fi password for a store using MCL?",
    requiredAny: [["cannot find", "could not find", "not in the", "not documented"]],
    forbidden: ["password is", "123", "admin"],
  },
];

function commonBrowserPaths() {
  return [
    process.env.EDGE_PATH,
    "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
    "C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe",
    "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
    "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",
  ].filter(Boolean);
}

async function launchBrowser() {
  try {
    return await chromium.launch({ headless: true });
  } catch (firstError) {
    for (const executablePath of commonBrowserPaths()) {
      if (!fs.existsSync(executablePath)) continue;
      try {
        return await chromium.launch({ headless: true, executablePath });
      } catch {
        // Try the next local Chromium-family browser.
      }
    }
    throw firstError;
  }
}

function includesAny(text, terms) {
  const lower = text.toLowerCase();
  return terms.some((term) => lower.includes(term.toLowerCase()));
}

function scoreAnswer(testCase, answer) {
  const answerBody = answer.split(/📚\s*(?:\*\*)?(?:Sources|Quellen)/i)[0];
  const requiredHits = (testCase.requiredAny || []).map((group) => ({
    group,
    pass: includesAny(answerBody, group),
  }));
  const forbiddenHits = (testCase.forbidden || []).filter((term) => includesAny(answerBody, [term]));
  return {
    pass: requiredHits.every((hit) => hit.pass) && forbiddenHits.length === 0,
    requiredHits,
    forbiddenHits,
  };
}

async function waitForAnswer(page, previousCount) {
  await page.locator(".thinking-indicator").waitFor({ state: "visible", timeout: 10000 }).catch(() => {});
  await page.locator(".thinking-indicator").waitFor({ state: "detached", timeout: TIMEOUT_MS }).catch(() => {});
  await page.waitForFunction(
    (count) => document.querySelectorAll(".message-bubble-container.assistant .message-content").length > count,
    previousCount,
    { timeout: TIMEOUT_MS },
  );
  const answers = await page.locator(".message-bubble-container.assistant .message-content").allInnerTexts();
  return answers[answers.length - 1] || "";
}

async function ask(page, prompt) {
  const beforeCount = await page.locator(".message-bubble-container.assistant .message-content").count();
  await page.locator("textarea.chat-textarea").fill(prompt);
  await page.locator("button.send-button").click();
  return await waitForAnswer(page, beforeCount);
}

async function renderedImageCount(page) {
  return await page.locator(".message-bubble-container.assistant img").evaluateAll((imgs) =>
    imgs.filter((img) => img.naturalWidth > 0 && img.naturalHeight > 0).length,
  );
}

async function main() {
  const browser = await launchBrowser();
  const page = await browser.newPage({
    ignoreHTTPSErrors: true,
    viewport: { width: 1440, height: 1000 },
  });

  const results = [];
  const errors = [];

  page.on("console", (msg) => {
    if (["warning", "error"].includes(msg.type())) {
      errors.push(`${msg.type()}: ${msg.text()}`);
    }
  });

  await page.goto(APP_URL, { waitUntil: "networkidle", timeout: 60000 });
  await page.locator("textarea.chat-textarea").waitFor({ state: "visible", timeout: 60000 });

  const newChat = page.locator('button[title="Start new conversation"]');
  if (await newChat.count()) {
    await newChat.click().catch(() => {});
    await page.waitForTimeout(1000);
  }

  for (const testCase of CASES) {
    const beforeImages = await renderedImageCount(page);
    let answer = "";
    let error = "";
    try {
      answer = await ask(page, testCase.prompt);
    } catch (err) {
      error = err && err.message ? err.message : String(err);
    }
    const afterImages = await renderedImageCount(page);
    const score = error ? { pass: false, requiredHits: [], forbiddenHits: [error] } : scoreAnswer(testCase, answer);
    const visualRendered = afterImages > beforeImages;
    results.push({ ...testCase, answer, error, ...score, visualRendered });
  }

  const failed = results.filter((result) => !result.pass);
  const highFailed = failed.filter((result) => result.priority === "high");

  console.log("# MCL Assistant Live Regression Report");
  console.log("");
  console.log(`App URL: ${APP_URL}`);
  console.log(`Passed: ${results.length - failed.length}/${results.length}`);
  console.log(`High priority failures: ${highFailed.length}`);
  console.log(`Console warnings/errors: ${errors.length}`);
  console.log("");

  for (const result of results) {
    console.log(`## ${result.pass ? "PASS" : "FAIL"} ${result.id} (${result.priority})`);
    console.log(`Prompt: ${result.prompt}`);
    if (result.error) console.log(`Error: ${result.error}`);
    console.log(`Forbidden hits: ${result.forbiddenHits.join(", ") || "none"}`);
    if (result.visualOptional) console.log(`Visual rendered in response: ${result.visualRendered}`);
    console.log("Answer excerpt:");
    console.log((result.answer || "").slice(0, 1200).replace(/\n{3,}/g, "\n\n"));
    console.log("");
  }

  if (errors.length) {
    console.log("## Console Warnings/Errors");
    for (const error of errors.slice(0, 20)) console.log(`- ${error}`);
    console.log("");
  }

  await browser.close();
  process.exit(highFailed.length ? 1 : 0);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
