import AppKit
import Foundation
import Darwin

final class DesktopApp: NSObject, NSApplicationDelegate {
    private let setupKey = "successfulSetup"
    private let chatURL = URL(string: "http://127.0.0.1:8000")!

    private var window: NSWindow!
    private var statusLabel: NSTextField!
    private var diskLabel: NSTextField!
    private var spinner: NSProgressIndicator!
    private var startButton: NSButton!
    private var openButton: NSButton!
    private var stopButton: NSButton!
    private var detailsButton: NSButton!
    private var copyButton: NSButton!
    private var logScroll: NSScrollView!
    private var logView: NSTextView!
    private var logLines: [String] = []
    private var outputBuffer = Data()
    private var supervisor: Process?
    private var inputPipe: Pipe?
    private var successfulSetup = false
    private var ready = false
    private var terminating = false

    func applicationDidFinishLaunching(_ notification: Notification) {
        installQuitMenu()
        successfulSetup = UserDefaults.standard.bool(forKey: setupKey)
        makeWindow()
        if successfulSetup {
            startSplash()
        }
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }

    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        guard let supervisor, supervisor.isRunning else { return .terminateNow }
        terminating = true
        sendAction("quit")
        inputPipe?.fileHandleForWriting.closeFile()
        DispatchQueue.main.asyncAfter(deadline: .now() + 25) { [weak self] in
            guard let self, self.supervisor?.isRunning == true else { return }
            self.supervisor?.terminate()
        }
        return .terminateLater
    }

    func applicationWillTerminate(_ notification: Notification) {
        inputPipe?.fileHandleForWriting.closeFile()
    }

    private func makeWindow() {
        window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 590, height: 520),
            styleMask: [.titled, .closable, .miniaturizable],
            backing: .buffered,
            defer: false
        )
        window.title = "Splash M1"
        window.center()

        let title = NSTextField(labelWithString: "Splash M1")
        title.font = .systemFont(ofSize: 25, weight: .semibold)

        let intro = NSTextField(
            wrappingLabelWithString:
                "First setup downloads the recommended Qwen3.8 model from Hugging Face. "
                + "Model weights are stored separately from the app, and cached files are reused. "
                + "Choose Download & Start to begin setup. Later launches may start automatically."
        )
        intro.maximumNumberOfLines = 4

        spinner = NSProgressIndicator()
        spinner.style = .spinning
        spinner.controlSize = .small
        spinner.isDisplayedWhenStopped = false
        spinner.translatesAutoresizingMaskIntoConstraints = false
        NSLayoutConstraint.activate([
            spinner.widthAnchor.constraint(equalToConstant: 16),
            spinner.heightAnchor.constraint(equalToConstant: 16),
        ])

        statusLabel = NSTextField(wrappingLabelWithString: "Ready to set up Splash.")
        statusLabel.maximumNumberOfLines = 3
        let statusRow = NSStackView(views: [spinner, statusLabel])
        statusRow.orientation = .horizontal
        statusRow.alignment = .centerY
        statusRow.spacing = 9

        diskLabel = NSTextField(labelWithString: "")
        diskLabel.textColor = .secondaryLabelColor

        startButton = NSButton(
            title: "Download & Start",
            target: self,
            action: #selector(startSplash)
        )
        startButton.keyEquivalent = "\r"
        openButton = NSButton(title: "Open Chat", target: self, action: #selector(openChat))
        openButton.isEnabled = false
        stopButton = NSButton(title: "Stop", target: self, action: #selector(stopSplash))
        stopButton.isEnabled = false

        let buttons = NSStackView(views: [startButton, openButton, stopButton])
        buttons.orientation = .horizontal
        buttons.alignment = .centerY
        buttons.spacing = 10

        detailsButton = NSButton(
            title: "Show Details",
            target: self,
            action: #selector(toggleDetails)
        )
        detailsButton.bezelStyle = .accessoryBarAction
        copyButton = NSButton(title: "Copy Output", target: self, action: #selector(copyOutput))
        copyButton.bezelStyle = .accessoryBarAction
        let detailsHeader = NSStackView(views: [detailsButton, copyButton])
        detailsHeader.orientation = .horizontal
        detailsHeader.spacing = 8

        logView = NSTextView()
        logView.isEditable = false
        logView.isSelectable = true
        logView.font = .monospacedSystemFont(ofSize: 11, weight: .regular)
        logView.textContainerInset = NSSize(width: 8, height: 8)
        logScroll = NSScrollView()
        logScroll.documentView = logView
        logScroll.hasVerticalScroller = true
        logScroll.borderType = .bezelBorder
        logScroll.translatesAutoresizingMaskIntoConstraints = false
        logScroll.isHidden = true
        logScroll.heightAnchor.constraint(equalToConstant: 180).isActive = true

        let stack = NSStackView(views: [title, intro, statusRow, diskLabel, buttons, detailsHeader, logScroll])
        stack.orientation = .vertical
        stack.alignment = .leading
        stack.distribution = .fill
        stack.spacing = 14
        stack.translatesAutoresizingMaskIntoConstraints = false

        let content = NSView()
        content.addSubview(stack)
        NSLayoutConstraint.activate([
            stack.leadingAnchor.constraint(equalTo: content.leadingAnchor, constant: 24),
            stack.trailingAnchor.constraint(equalTo: content.trailingAnchor, constant: -24),
            stack.topAnchor.constraint(equalTo: content.topAnchor, constant: 24),
            stack.bottomAnchor.constraint(lessThanOrEqualTo: content.bottomAnchor, constant: -24),
            logScroll.widthAnchor.constraint(equalTo: stack.widthAnchor),
            statusLabel.widthAnchor.constraint(greaterThanOrEqualToConstant: 360),
            intro.widthAnchor.constraint(equalTo: stack.widthAnchor),
        ])
        window.contentView = content
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    private func installQuitMenu() {
        let mainMenu = NSMenu()
        let appMenuItem = NSMenuItem()
        let appMenu = NSMenu()
        let quitItem = NSMenuItem(
            title: "Quit Splash M1",
            action: #selector(NSApplication.terminate(_:)),
            keyEquivalent: "q"
        )
        quitItem.target = NSApp
        appMenu.addItem(quitItem)
        appMenuItem.submenu = appMenu
        mainMenu.addItem(appMenuItem)
        NSApp.mainMenu = mainMenu
    }

    @objc private func startSplash() {
        if let supervisor, supervisor.isRunning {
            setStarting()
            sendAction("start")
            return
        }

        guard let resources = Bundle.main.resourceURL else {
            showError("The app bundle has no Resources folder.")
            return
        }
        let runtime = resources.appendingPathComponent("runtime", isDirectory: true)
        let python = runtime.appendingPathComponent("python/bin/python3")
        let helper = runtime.appendingPathComponent("install/desktop.py")
        guard FileManager.default.isExecutableFile(atPath: python.path),
              FileManager.default.fileExists(atPath: helper.path) else {
            showError("The bundled Splash runtime is incomplete. Reinstall the app and retry.")
            return
        }

        let process = Process()
        let input = Pipe()
        let output = Pipe()
        process.executableURL = python
        process.arguments = ["-u", helper.path]
        process.currentDirectoryURL = runtime
        process.standardInput = input
        process.standardOutput = output
        process.standardError = output
        process.environment = ProcessInfo.processInfo.environment.merging(
            ["PYTHONDONTWRITEBYTECODE": "1"],
            uniquingKeysWith: { _, new in new }
        )
        process.terminationHandler = { [weak self] child in
            DispatchQueue.main.async { self?.supervisorExited(child) }
        }

        do {
            try process.run()
        } catch {
            showError("Could not start Splash: \(error.localizedDescription)")
            return
        }
        supervisor = process
        inputPipe = input
        outputBuffer.removeAll(keepingCapacity: true)
        output.fileHandleForReading.readabilityHandler = { [weak self, weak process] handle in
            let data = handle.availableData
            guard !data.isEmpty else {
                handle.readabilityHandler = nil
                return
            }
            DispatchQueue.main.async {
                guard let self, let process, self.supervisor === process else { return }
                self.receive(data)
            }
        }
        setStarting()
        sendAction("start")
    }

    @objc private func stopSplash() {
        guard supervisor?.isRunning == true else { return }
        statusLabel.stringValue = "Stopping Splash…"
        stopButton.isEnabled = false
        sendAction("stop")
    }

    @objc private func openChat() {
        NSWorkspace.shared.open(chatURL)
    }

    @objc private func toggleDetails() {
        logScroll.isHidden.toggle()
        detailsButton.title = logScroll.isHidden ? "Show Details" : "Hide Details"
        window.contentView?.layoutSubtreeIfNeeded()
    }

    @objc private func copyOutput() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(logView.string, forType: .string)
    }

    private func setStarting() {
        ready = false
        spinner.startAnimation(nil)
        statusLabel.stringValue = "Starting Splash…"
        startButton.isEnabled = false
        startButton.title = successfulSetup ? "Start Splash" : "Download & Start"
        openButton.isEnabled = false
        stopButton.isEnabled = true
    }

    private func receive(_ data: Data) {
        outputBuffer.append(data)
        while let newline = outputBuffer.firstRange(of: Data([0x0A])) {
            let lineData = outputBuffer.subdata(in: outputBuffer.startIndex..<newline.lowerBound)
            outputBuffer.removeSubrange(outputBuffer.startIndex..<newline.upperBound)
            handleLine(String(decoding: lineData, as: UTF8.self))
        }
    }

    private func handleLine(_ line: String) {
        guard let data = line.data(using: .utf8),
              let event = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
              let type = event["type"] as? String else {
            appendLog(line)
            return
        }
        switch type {
        case "status":
            statusLabel.stringValue = event["message"] as? String ?? "Working…"
        case "disk":
            if let bytes = event["available_bytes"] as? Int64 {
                let formatter = ByteCountFormatter()
                formatter.allowedUnits = [.useGB, .useTB]
                formatter.countStyle = .file
                diskLabel.stringValue = "Available disk space: \(formatter.string(fromByteCount: bytes))"
            }
        case "log":
            let message = event["message"] as? String ?? ""
            appendLog(message)
            updateProgressFromLog(message)
        case "ready":
            ready = true
            successfulSetup = true
            UserDefaults.standard.set(true, forKey: setupKey)
            spinner.stopAnimation(nil)
            statusLabel.stringValue = "Splash is ready. Opening chat…"
            startButton.isEnabled = false
            openButton.isEnabled = true
            stopButton.isEnabled = true
            NSWorkspace.shared.open(chatURL)
        case "error":
            ready = false
            spinner.stopAnimation(nil)
            statusLabel.stringValue = event["message"] as? String ?? "Splash could not start."
            startButton.title = successfulSetup ? "Retry" : "Retry Setup"
            startButton.isEnabled = true
            openButton.isEnabled = false
            stopButton.isEnabled = false
        case "stopped":
            ready = false
            spinner.stopAnimation(nil)
            statusLabel.stringValue = "Splash is stopped."
            startButton.title = successfulSetup ? "Start Splash" : "Download & Start"
            startButton.isEnabled = true
            openButton.isEnabled = false
            stopButton.isEnabled = false
        default:
            appendLog(line)
        }
    }

    private func appendLog(_ line: String) {
        let bounded = String(line.prefix(4096))
        logLines.append(bounded)
        if logLines.count > 300 {
            logLines.removeFirst(logLines.count - 300)
        }
        while logLines.joined(separator: "\n").utf8.count > 65_536 && logLines.count > 1 {
            logLines.removeFirst()
        }
        logView.string = logLines.joined(separator: "\n")
        logView.scrollToEndOfDocument(nil)
    }

    private func updateProgressFromLog(_ line: String) {
        guard !ready else { return }
        let value = line.trimmingCharacters(in: .whitespacesAndNewlines)
        let lower = value.lowercased()
        if value.hasPrefix("Fetching ") {
            statusLabel.stringValue = String(value.prefix(180))
        } else if lower.contains("downloading")
            || (lower.contains("safetensors") && lower.contains("%"))
        {
            statusLabel.stringValue = String(value.prefix(180))
        } else if lower.contains("prepar") || lower.contains("verif") || lower.contains("install") {
            statusLabel.stringValue = String(value.prefix(180))
        } else if value.contains("Loading ·") {
            statusLabel.stringValue = "Loading the model into memory…"
        } else if value.contains("Ready ·") {
            statusLabel.stringValue = "Waiting for Splash to become ready…"
        }
    }

    private func sendAction(_ action: String) {
        guard let inputPipe else { return }
        do {
            var data = try JSONSerialization.data(withJSONObject: ["action": action])
            data.append(0x0A)
            try inputPipe.fileHandleForWriting.write(contentsOf: data)
        } catch {
            showError("Could not communicate with Splash: \(error.localizedDescription)")
        }
    }

    private func showError(_ message: String) {
        spinner?.stopAnimation(nil)
        statusLabel?.stringValue = message
        startButton?.isEnabled = true
        openButton?.isEnabled = false
        stopButton?.isEnabled = false
    }

    private func supervisorExited(_ child: Process) {
        guard supervisor === child else { return }
        if terminating {
            NSApp.reply(toApplicationShouldTerminate: true)
            return
        }
        supervisor = nil
        inputPipe = nil
        let wasReady = ready
        ready = false
        spinner.stopAnimation(nil)
        statusLabel.stringValue = wasReady
            ? "Splash stopped unexpectedly. Retry to start it again."
            : "Splash setup stopped unexpectedly (code \(child.terminationStatus)). Check Details, then retry."
        startButton.title = "Retry"
        startButton.isEnabled = true
        openButton.isEnabled = false
        stopButton.isEnabled = false
    }

    static func checkBundle() -> Int32 {
        guard let resources = Bundle.main.resourceURL else {
            FileHandle.standardError.write(Data("Missing app Resources folder.\n".utf8))
            return 1
        }
        let runtime = resources.appendingPathComponent("runtime", isDirectory: true)
        let executablePaths = [
            runtime.appendingPathComponent("python/bin/python3"),
            runtime.appendingPathComponent("engine/splash"),
        ]
        let requiredPaths = [
            runtime.appendingPathComponent("release.json"),
            runtime.appendingPathComponent("install/desktop.py"),
            runtime.appendingPathComponent("install/launcher.py"),
            runtime.appendingPathComponent("engine/splash.metallib"),
        ] + executablePaths
        let missing = requiredPaths.filter { !FileManager.default.fileExists(atPath: $0.path) }
        let notExecutable = executablePaths.filter {
            !FileManager.default.isExecutableFile(atPath: $0.path)
        }
        guard missing.isEmpty, notExecutable.isEmpty else {
            let details = (missing + notExecutable).map(\.path).joined(separator: ", ")
            FileHandle.standardError.write(Data("Incomplete bundled runtime: \(details)\n".utf8))
            return 1
        }
        print("Splash M1 bundle OK")
        return 0
    }

    static func runApplication() {
        let application = NSApplication.shared
        let delegate = DesktopApp()
        application.setActivationPolicy(.regular)
        application.delegate = delegate
        application.run()
    }
}

if CommandLine.arguments.contains("--check") {
    exit(DesktopApp.checkBundle())
}
DesktopApp.runApplication()
