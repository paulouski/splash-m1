import AppKit
import Foundation
import Darwin

final class DesktopApp: NSObject, NSApplicationDelegate {
    private let setupKey = "successfulSetup"
    private let selectedModelKey = "selectedModel"
    private let recommendedModel = "mlx-community/Qwen3.8-27B-4bit"
    private let chatURL = URL(string: "http://127.0.0.1:8000")!
    private let apiURL = "http://127.0.0.1:8000/v1"

    private var window: NSWindow!
    private var statusLabel: NSTextField!
    private var diskLabel: NSTextField!
    private var spinner: NSProgressIndicator!
    private var startButton: NSButton!
    private var openButton: NSButton!
    private var stopButton: NSButton!
    private var modelInputField: NSTextField!
    private var checkModelButton: NSButton!
    private var installedModelsPicker: NSPopUpButton!
    private var refreshModelsButton: NSButton!
    private var deleteModelButton: NSButton!
    private var detailsButton: NSButton!
    private var copyButton: NSButton!
    private var copyPiButton: NSButton!
    private var copyOpenCodeButton: NSButton!
    private var copyAPIButton: NSButton!
    private var logScroll: NSScrollView!
    private var logView: NSTextView!
    private var logLines: [String] = []
    private var outputBuffer = Data()
    private var supervisor: Process?
    private var inputPipe: Pipe?
    private var successfulSetup = false
    private var ready = false
    private var isStarting = false
    private var isCheckingModel = false
    private var isStopping = false
    private var terminating = false

    func applicationDidFinishLaunching(_ notification: Notification) {
        installQuitMenu()
        successfulSetup = UserDefaults.standard.bool(forKey: setupKey)
        makeWindow()
        let helperStarted = startSupervisor()
        if helperStarted {
            sendAction("models")
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
            contentRect: NSRect(x: 0, y: 0, width: 590, height: 630),
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
                "Choose an installed model or enter a Hugging Face repo ID or URL. "
                + "Model files are stored separately from the app, and cached files are reused. "
                + "Later launches may start the last model automatically."
        )
        intro.maximumNumberOfLines = 4

        modelInputField = NSTextField(string: selectedModel())
        modelInputField.placeholderString = "Hugging Face repo ID or URL"
        modelInputField.target = self
        modelInputField.action = #selector(modelInputChanged)
        modelInputField.setContentHuggingPriority(.defaultLow, for: .horizontal)
        modelInputField.setContentCompressionResistancePriority(.defaultLow, for: .horizontal)
        checkModelButton = NSButton(
            title: "Check Model",
            target: self,
            action: #selector(checkModel)
        )
        let modelEntryRow = NSStackView(views: [modelInputField, checkModelButton])
        modelEntryRow.orientation = .horizontal
        modelEntryRow.alignment = .centerY
        modelEntryRow.spacing = 8

        installedModelsPicker = NSPopUpButton()
        installedModelsPicker.addItem(withTitle: "Installed models")
        installedModelsPicker.lastItem?.isEnabled = false
        installedModelsPicker.target = self
        installedModelsPicker.action = #selector(selectInstalledModel)
        installedModelsPicker.setContentHuggingPriority(.defaultLow, for: .horizontal)
        installedModelsPicker.setContentCompressionResistancePriority(.defaultLow, for: .horizontal)
        refreshModelsButton = NSButton(
            title: "Refresh",
            target: self,
            action: #selector(refreshInstalledModels)
        )
        deleteModelButton = NSButton(
            title: "Delete…",
            target: self,
            action: #selector(deleteInstalledModel)
        )
        let installedModelsRow = NSStackView(
            views: [installedModelsPicker, refreshModelsButton, deleteModelButton]
        )
        installedModelsRow.orientation = .horizontal
        installedModelsRow.alignment = .centerY
        installedModelsRow.spacing = 8

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

        let connectionInstructions = NSTextField(
            wrappingLabelWithString:
                "Keep Splash M1 running. Install Pi or OpenCode, then paste its command into Terminal "
                + "opened in your project folder. The commands connect the client to the running model. "
                + "Pi adds a Splash provider; OpenCode settings apply to that launch."
        )
        connectionInstructions.maximumNumberOfLines = 3
        let connectionTitle = NSTextField(labelWithString: "Connect a coding client")
        connectionTitle.font = .systemFont(ofSize: 13, weight: .semibold)
        copyPiButton = NSButton(
            title: "Copy Pi Command",
            target: self,
            action: #selector(copyPiCommand)
        )
        copyOpenCodeButton = NSButton(
            title: "Copy OpenCode Command",
            target: self,
            action: #selector(copyOpenCodeCommand)
        )
        copyAPIButton = NSButton(
            title: "Copy API URL",
            target: self,
            action: #selector(copyAPIURL)
        )
        let connectionButtons = NSStackView(views: [copyPiButton, copyOpenCodeButton, copyAPIButton])
        connectionButtons.orientation = .horizontal
        connectionButtons.alignment = .centerY
        connectionButtons.spacing = 8

        logScroll = NSScrollView(frame: NSRect(x: 0, y: 0, width: 542, height: 180))
        logView = NSTextView(frame: NSRect(origin: .zero, size: logScroll.contentSize))
        logView.isEditable = false
        logView.isSelectable = true
        logView.font = .monospacedSystemFont(ofSize: 11, weight: .regular)
        logView.textContainerInset = NSSize(width: 8, height: 8)
        logView.minSize = NSSize(width: 0, height: 0)
        logView.maxSize = NSSize(
            width: CGFloat.greatestFiniteMagnitude,
            height: CGFloat.greatestFiniteMagnitude
        )
        logView.isVerticallyResizable = true
        logView.isHorizontallyResizable = false
        logView.autoresizingMask = [.width]
        logView.textContainer?.containerSize = NSSize(
            width: logScroll.contentSize.width,
            height: CGFloat.greatestFiniteMagnitude
        )
        logView.textContainer?.widthTracksTextView = true
        logScroll.documentView = logView
        logScroll.hasVerticalScroller = true
        logScroll.borderType = .bezelBorder
        logScroll.translatesAutoresizingMaskIntoConstraints = false
        logScroll.isHidden = true
        logScroll.heightAnchor.constraint(equalToConstant: 180).isActive = true

        let stack = NSStackView(views: [
            title,
            intro,
            modelEntryRow,
            installedModelsRow,
            statusRow,
            diskLabel,
            buttons,
            connectionTitle,
            connectionInstructions,
            connectionButtons,
            detailsHeader,
            logScroll,
        ])
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
            modelEntryRow.widthAnchor.constraint(equalTo: stack.widthAnchor),
            installedModelsRow.widthAnchor.constraint(equalTo: stack.widthAnchor),
            connectionInstructions.widthAnchor.constraint(equalTo: stack.widthAnchor),
        ])
        window.contentView = content
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
        updateModelControls()
        updateConnectionControls()
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
        let model = modelInputField.stringValue.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !model.isEmpty else {
            statusLabel.stringValue = "Enter a Hugging Face model ID or URL."
            return
        }
        guard startSupervisor() else { return }
        setStarting()
        sendAction("start", model: model)
    }

    private func startSupervisor() -> Bool {
        if let supervisor, supervisor.isRunning { return true }
        guard let resources = Bundle.main.resourceURL else {
            showError("The app bundle has no Resources folder.")
            return false
        }
        let runtime = resources.appendingPathComponent("runtime", isDirectory: true)
        let python = runtime.appendingPathComponent("python/bin/python3")
        let helper = runtime.appendingPathComponent("install/desktop.py")
        guard FileManager.default.isExecutableFile(atPath: python.path),
              FileManager.default.fileExists(atPath: helper.path) else {
            showError("The bundled Splash runtime is incomplete. Reinstall the app and retry.")
            return false
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
            return false
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
        return true
    }

    @objc private func checkModel() {
        let model = modelInputField.stringValue.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !model.isEmpty else {
            statusLabel.stringValue = "Enter a Hugging Face model ID or URL."
            return
        }
        guard startSupervisor() else { return }
        ready = false
        isStarting = false
        isCheckingModel = true
        isStopping = false
        spinner.startAnimation(nil)
        statusLabel.stringValue = "Checking model metadata…"
        startButton.isEnabled = false
        openButton.isEnabled = false
        stopButton.isEnabled = true
        updateModelControls()
        updateConnectionControls()
        sendAction("check", model: model)
    }

    @objc private func refreshInstalledModels() {
        guard startSupervisor() else { return }
        sendAction("models")
    }

    @objc private func deleteInstalledModel() {
        guard let model = installedModelsPicker.selectedItem?.representedObject as? String,
              startSupervisor() else { return }
        sendAction("delete_plan", model: model)
    }

    private func confirmDelete(_ model: String, bytes: Int64) {
        let formatter = ByteCountFormatter()
        formatter.countStyle = .file
        let alert = NSAlert()
        alert.messageText = "Delete \(model)?"
        alert.informativeText =
            "Frees about \(formatter.string(fromByteCount: bytes)). The DFlash draft is kept."
        alert.alertStyle = .warning
        alert.addButton(withTitle: "Delete")
        alert.addButton(withTitle: "Cancel")
        if alert.runModal() == .alertFirstButtonReturn {
            sendAction("delete", model: model)
        }
    }

    @objc private func modelInputChanged() {
        updateModelControls()
    }

    @objc private func selectInstalledModel() {
        guard let model = installedModelsPicker.selectedItem?.representedObject as? String else {
            return
        }
        modelInputField.stringValue = model
        updateModelControls()
    }

    @objc private func stopSplash() {
        guard supervisor?.isRunning == true else { return }
        isStopping = true
        statusLabel.stringValue = "Stopping Splash…"
        stopButton.isEnabled = false
        updateModelControls()
        updateConnectionControls()
        sendAction("stop")
    }

    @objc private func openChat() {
        NSWorkspace.shared.open(chatURL)
    }

    @objc private func toggleDetails() {
        let showingDetails = logScroll.isHidden
        logScroll.isHidden = !showingDetails
        detailsButton.title = logScroll.isHidden ? "Show Details" : "Hide Details"
        var frame = window.frame
        let heightChange: CGFloat = showingDetails ? 180 : -180
        frame.size.height += heightChange
        frame.origin.y -= heightChange / 2
        window.setFrame(frame, display: true, animate: true)
        window.contentView?.layoutSubtreeIfNeeded()
    }

    @objc private func copyOutput() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(logView.string, forType: .string)
    }

    @objc private func copyPiCommand() {
        copyClientCommand("pi")
    }

    @objc private func copyOpenCodeCommand() {
        copyClientCommand("opencode")
    }

    @objc private func copyAPIURL() {
        guard ready, !isStopping else { return }
        copyToPasteboard(apiURL)
    }

    private func copyClientCommand(_ client: String) {
        guard ready, !isStopping,
              let resources = Bundle.main.resourceURL else { return }
        let launcher = resources
            .appendingPathComponent("runtime", isDirectory: true)
            .appendingPathComponent("splash-m1")
            .path
        copyToPasteboard("\(shellQuote(launcher)) \(client)")
    }

    private func shellQuote(_ value: String) -> String {
        "'\(value.replacingOccurrences(of: "'", with: "'\\''"))'"
    }

    private func copyToPasteboard(_ value: String) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(value, forType: .string)
    }

    private func selectedModel() -> String {
        UserDefaults.standard.string(forKey: selectedModelKey) ?? recommendedModel
    }

    private func updateModelControls() {
        guard modelInputField != nil else { return }
        let enabled = !ready && !isStarting && !isCheckingModel && !isStopping
        modelInputField.isEnabled = enabled
        checkModelButton.isEnabled = enabled
        refreshModelsButton.isEnabled = enabled
        deleteModelButton.isEnabled =
            enabled && installedModelsPicker.selectedItem?.representedObject is String
        installedModelsPicker.isEnabled = enabled && installedModelsPicker.numberOfItems > 1
    }

    private func updateConnectionControls() {
        let enabled = ready && !isStopping
        copyPiButton?.isEnabled = enabled
        copyOpenCodeButton?.isEnabled = enabled
        copyAPIButton?.isEnabled = enabled
    }

    private func updateInstalledModels(_ values: [[String: Any]]) {
        installedModelsPicker.removeAllItems()
        let models = values.compactMap { value -> (String, String)? in
            guard let model = value["model"] as? String, !model.isEmpty else { return nil }
            let name = (value["name"] as? String ?? "")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            let displayName = name.isEmpty ? model : name
            return (model, displayName)
        }
        if models.isEmpty {
            installedModelsPicker.addItem(withTitle: "No installed models")
            installedModelsPicker.lastItem?.isEnabled = false
        } else {
            installedModelsPicker.addItem(withTitle: "Installed models")
            installedModelsPicker.lastItem?.isEnabled = false
            for (model, name) in models {
                installedModelsPicker.addItem(withTitle: name)
                installedModelsPicker.lastItem?.representedObject = model
            }
        }
        updateModelControls()
    }

    private func setStarting() {
        ready = false
        isStarting = true
        isCheckingModel = false
        isStopping = false
        spinner.startAnimation(nil)
        statusLabel.stringValue = "Starting Splash…"
        startButton.isEnabled = false
        startButton.title = successfulSetup ? "Start Splash" : "Download & Start"
        openButton.isEnabled = false
        stopButton.isEnabled = true
        updateModelControls()
        updateConnectionControls()
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
        case "models":
            updateInstalledModels(event["models"] as? [[String: Any]] ?? [])
        case "delete_plan":
            if let model = event["model"] as? String {
                confirmDelete(model, bytes: (event["bytes"] as? NSNumber)?.int64Value ?? 0)
            }
        case "deleted":
            let model = event["model"] as? String
            if model == selectedModel() {
                UserDefaults.standard.set(recommendedModel, forKey: selectedModelKey)
            }
            if model == modelInputField.stringValue {
                modelInputField.stringValue = recommendedModel
            }
            let remaining = (0..<installedModelsPicker.numberOfItems).filter {
                let item = installedModelsPicker.item(at: $0)
                return item?.representedObject is String && item?.representedObject as? String != model
            }
            if remaining.isEmpty {
                successfulSetup = false
                UserDefaults.standard.set(false, forKey: setupKey)
                startButton.title = "Download & Start"
            }
            statusLabel.stringValue = "Deleted \(model ?? "model")."
        case "model_checked":
            isCheckingModel = false
            spinner.stopAnimation(nil)
            if let model = event["model"] as? String {
                modelInputField.stringValue = model
            }
            statusLabel.stringValue = event["message"] as? String
                ?? "Model metadata is compatible; native tensor validation occurs during startup."
            startButton.title = successfulSetup ? "Start Splash" : "Download & Start"
            startButton.isEnabled = true
            openButton.isEnabled = false
            stopButton.isEnabled = false
            updateModelControls()
            updateConnectionControls()
        case "ready":
            ready = true
            isStarting = false
            isCheckingModel = false
            isStopping = false
            successfulSetup = true
            UserDefaults.standard.set(true, forKey: setupKey)
            let model = event["model"] as? String ?? modelInputField.stringValue
            modelInputField.stringValue = model
            UserDefaults.standard.set(model, forKey: selectedModelKey)
            spinner.stopAnimation(nil)
            statusLabel.stringValue = "Splash is ready."
            startButton.isEnabled = false
            openButton.isEnabled = true
            stopButton.isEnabled = true
            updateModelControls()
            updateConnectionControls()
        case "error":
            ready = false
            isStarting = false
            isCheckingModel = false
            isStopping = false
            spinner.stopAnimation(nil)
            statusLabel.stringValue = event["message"] as? String ?? "Splash could not start."
            startButton.title = successfulSetup ? "Retry" : "Retry Setup"
            startButton.isEnabled = true
            openButton.isEnabled = false
            stopButton.isEnabled = false
            updateModelControls()
            updateConnectionControls()
        case "stopped":
            ready = false
            isStarting = false
            isCheckingModel = false
            isStopping = false
            spinner.stopAnimation(nil)
            statusLabel.stringValue = "Splash is stopped."
            startButton.title = successfulSetup ? "Start Splash" : "Download & Start"
            startButton.isEnabled = true
            openButton.isEnabled = false
            stopButton.isEnabled = false
            updateModelControls()
            updateConnectionControls()
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

    private func sendAction(_ action: String, model: String? = nil) {
        guard let inputPipe else { return }
        do {
            var command: [String: String] = ["action": action]
            if let model { command["model"] = model }
            var data = try JSONSerialization.data(withJSONObject: command)
            data.append(0x0A)
            try inputPipe.fileHandleForWriting.write(contentsOf: data)
        } catch {
            showError("Could not communicate with Splash: \(error.localizedDescription)")
        }
    }

    private func showError(_ message: String) {
        ready = false
        isStarting = false
        isCheckingModel = false
        isStopping = false
        spinner?.stopAnimation(nil)
        statusLabel?.stringValue = message
        startButton?.isEnabled = true
        openButton?.isEnabled = false
        stopButton?.isEnabled = false
        updateModelControls()
        updateConnectionControls()
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
        isStarting = false
        isCheckingModel = false
        isStopping = false
        spinner.stopAnimation(nil)
        statusLabel.stringValue = wasReady
            ? "Splash stopped unexpectedly. Retry to start it again."
            : "Splash setup stopped unexpectedly (code \(child.terminationStatus)). Check Details, then retry."
        startButton.title = "Retry"
        startButton.isEnabled = true
        openButton.isEnabled = false
        stopButton.isEnabled = false
        updateModelControls()
        updateConnectionControls()
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
            runtime.appendingPathComponent("install/desktop_models.py"),
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
