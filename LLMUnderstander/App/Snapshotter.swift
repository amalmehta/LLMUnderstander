import SwiftUI
import AppKit

/// Developer tool: `LLM Understander --snapshots <folder>` trains the tiny model,
/// visits every chapter and saves a PNG of the window (used for the README).
enum Snapshotter {
    @MainActor
    static func runIfRequested(_ app: AppState) async {
        let args = CommandLine.arguments
        guard let i = args.firstIndex(of: "--snapshots"), i + 1 < args.count else { return }
        let dir = URL(fileURLWithPath: args[i + 1])
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try? await Task.sleep(for: .seconds(1))
        // `--tall` makes the window taller than the screen so whole chapters fit (for reviewing).
        let height = args.contains("--tall") ? 3200.0 : 880.0
        if let w = NSApp.windows.first {
            w.setContentSize(NSSize(width: 1320, height: height))
            w.setFrameTopLeftPoint(NSPoint(x: 40, y: (NSScreen.main?.frame.maxY ?? 900) - 30))
        }
        NSApp.activate(ignoringOtherApps: true)
        // Exercise the real background training path first, then finish synchronously.
        app.trainer.start()
        try? await Task.sleep(for: .seconds(4))
        app.trainer.pause()
        try? await Task.sleep(for: .seconds(0.5))
        app.trainer.trainNow(steps: 1200 - app.trainer.step)
        for c in Chapter.allCases {
            app.selected = c
            try? await Task.sleep(for: .seconds(2.5))
            save(to: dir.appendingPathComponent(String(format: "%02d %@.png", c.number, c.title)))
        }
        NSApp.terminate(nil)
    }

    /// Ask the system screenshot tool for just this app's window (works even if other windows overlap it).
    @MainActor
    private static func save(to url: URL) {
        guard let window = NSApp.windows.first(where: { $0.isVisible && $0.contentView != nil && $0.frame.width > 600 }) else { return }
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/usr/sbin/screencapture")
        p.arguments = ["-x", "-o", "-l", "\(window.windowNumber)", url.path]
        try? p.run()
        p.waitUntilExit()
    }
}
