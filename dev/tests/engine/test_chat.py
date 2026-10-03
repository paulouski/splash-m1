import shutil
import subprocess
import unittest
from pathlib import Path

NODE = shutil.which("node")
CHAT = Path(__file__).resolve().parents[3] / "server" / "chat.html"

# Execute the complete production script with a minimal DOM and controlled I/O.
# Tests drive the registered UI handlers; no copied production handler or network.
HARNESS = r"""
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const script = fs.readFileSync(process.argv[1], 'utf8')
  .match(/<script>([\s\S]*?)<\/script>/)[1];

class Element {
  constructor(tag = 'div') {
    this.tagName = tag.toUpperCase();
    this.handlers = {};
    this.value = '';
    this.style = {};
    this.options = [];
    this.children = [];
    this.scrollHeight = 40;
    this.classList = {add() {}, remove() {}, toggle() {}};
  }
  addEventListener(name, callback) { this.handlers[name] = callback; }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = children; }
  remove() {} focus() {} setAttribute() {} removeAttribute() {}
  querySelector(selector) { return (this.found ||= {})[selector] ||= new Element(); }
}

function createChat(storage = new Map(), writable = true, models = null,
                    crypto = require('node:crypto').webcrypto, status = null, settings = null) {
  const elements = {};
  const requests = [];
  const modelRequests = [];
  const reads = [];
  const statusRequests = [];
  const settingsRequests = [];
  const toolRequests = [];
  let tool = null;
  const view = {y: 0, height: 1000, inner: 800};
  const scrolls = [];
  const windowHandlers = {};
  const clock = {now: 1e12};
  const intervals = new Map();
  let nextInterval = 1;
  for (const id of ['chat', 'form', 'input', 'attachments', 'image-input',
                   'attach', 'effort', 'web', 'api-key', 'send', 'recents', 'new-chat',
                   'mobile-new', 'menu', 'scrim', 'jump', 'model-dot', 'model-text',
                   'model-sub', 'idle-select', 'ctx', 'ctx-text', 'ctx-fill',
                   'gear', 'gear-label', 'settings', 's-system', 's-system-hint', 's-system-default',
                   'preset-precise', 'preset-balance', 'preset-creative', 'preset-reset',
                   's-temperature', 's-temperature-n', 's-top_p', 's-top_p-n', 's-top_k', 's-top_k-n',
                   's-max_tokens-n', 's-seed-n']) elements[id] = new Element();
  elements.effort.options = ['', 'xhigh', 'medium', 'low', 'none'].map(value => ({value}));
  class FileReader {
    readAsDataURL(file) {
      this.result = `data:image/png;base64,${file.name}`;
      reads.push(this);
    }
  }
  const context = vm.createContext({
    document: {
      querySelector: selector => elements[selector.slice(1)] || null,
      createElement: tag => new Element(tag),
      body: new Element(),
      documentElement: {get scrollHeight() { return view.height; }},
      addEventListener() {},
      hidden: false,
    },
    get scrollY() { return view.y; },
    get innerHeight() { return view.inner; },
    addEventListener(name, callback) { windowHandlers[name] = callback; },
    localStorage: {
      getItem: key => storage.get(key) ?? null,
      setItem(key, value) {
        if (!writable) throw new Error('Storage is full');
        storage.set(key, value);
      },
    },
    scrollTo(x, y) { scrolls.push(y); view.y = Math.min(y, view.height - view.inner); windowHandlers.scroll?.(); },
    __clock: clock,
    setInterval(callback) { intervals.set(nextInterval, callback); return nextInterval++; },
    clearInterval(id) { intervals.delete(id); },
    AbortController, TextDecoder, Uint8Array, console, FileReader, URL,
    crypto,
    fetch(url, options) {
      if (url === '/v1/models') {
        modelRequests.push(options.headers);
        // models answers from the request headers; without it the server is offline.
        return models ? Promise.resolve(models(options.headers))
          : Promise.reject(new Error('offline'));
      }
      if (url === '/status') {
        statusRequests.push(options.headers);
        return status ? Promise.resolve(status(options.headers)) : Promise.reject(new Error('offline'));
      }
      if (url === '/splash/settings') {
        settingsRequests.push({method: options.method || 'GET', headers: options.headers, body: options.body && JSON.parse(options.body)});
        return settings ? Promise.resolve(settings(options)) : Promise.reject(new Error('offline'));
      }
      if (url.startsWith('/splash/tools/')) {
        const body = JSON.parse(options.body);
        toolRequests.push({url, body, headers: options.headers});
        const answer = tool ? tool(url, body) : 'hang';
        // 'hang' waits until the loop is stopped.
        if (answer !== 'hang') return Promise.resolve(answer);
        return new Promise((resolve, reject) => options.signal.addEventListener('abort', () => {
          reject(Object.assign(new Error('Stopped'), {name: 'AbortError'}));
        }));
      }
      return new Promise((resolve, reject) => {
        requests.push({url, headers: options.headers, body: JSON.parse(options.body), resolve, reject});
        options.signal.addEventListener('abort', () => {
          reject(Object.assign(new Error('Stopped'), {name: 'AbortError'}));
        });
      });
    },
  });
  vm.runInContext('Date.now = () => __clock.now++', context);
  vm.runInContext(script, context);
  // Runs every timer once, as time passing would.
  const tick = ms => { clock.now += ms; [...intervals.values()].forEach(callback => callback()); };
  return {set tool(handler) { tool = handler; }, toolRequests, clock, tick, elements, requests, reads, modelRequests, statusRequests, settingsRequests, view, scrolls, windowHandlers, context};
}

const flush = () => new Promise(resolve => setImmediate(resolve));
const submit = chat => chat.elements.form.handlers.submit({preventDefault() {}});
const enter = chat => chat.elements.input.handlers.keydown({
  key: 'Enter', keyCode: 13, shiftKey: false, isComposing: false,
  preventDefault() {},
});
function setText(chat, value) {
  chat.elements.input.value = value;
  chat.elements.input.handlers.input();
}
function pasteImages(chat, ...names) {
  const start = chat.reads.length;
  chat.elements.input.handlers.paste({
    clipboardData: {
      items: names.map(name => ({type: 'image/png', getAsFile: () => ({type: 'image/png', name})})),
      getData: () => '',
    },
    preventDefault() {},
  });
  return chat.reads.slice(start);
}
const pasteImage = (chat, name) => pasteImages(chat, name)[0];
function selectImages(chat, ...names) {
  const start = chat.reads.length;
  chat.elements['image-input'].files = names.map(name => ({type: 'image/png', name}));
  chat.elements['image-input'].handlers.change();
  return chat.reads.slice(start);
}
const imagePart = name => ({type: 'image_url', image_url: {url: `data:image/png;base64,${name}`}});
const draftImages = chat => chat.elements.attachments.children.flatMap(
  item => item.children.filter(child => child.tagName === 'IMG').map(image => image.src)
);
function removeImage(chat, index) {
  const item = chat.elements.attachments.children[index];
  item.children.find(child => child.tagName === 'BUTTON').handlers.click();
}

const sse = chunks => Buffer.from(chunks.map(chunk => `data: ${JSON.stringify(chunk)}\n\n`).join('') + 'data: [DONE]\n\n');
function respond(request, chunks) {
  let sent = false;
  request.resolve({ok: true, body: {getReader: () => ({
    async read() {
      if (sent) return {done: true};
      sent = true;
      return {done: false, value: sse(chunks)};
    },
  })}});
}
// A streamed tool call whose arguments arrive in two pieces, as the server sends them.
const callChunks = (id, name, args) => {
  const json = JSON.stringify(args);
  const delta = call => ({choices: [{delta: {tool_calls: [call]}}]});
  return [
    delta({index: 0, id, type: 'function', function: {name}}),
    delta({index: 0, function: {arguments: json.slice(0, 6)}}),
    delta({index: 0, function: {arguments: json.slice(6)}}),
    {choices: [{delta: {}, finish_reason: 'tool_calls'}]},
    {choices: [], usage: {prompt_tokens: 50, completion_tokens: 5}},
  ];
};
const answerChunks = text => [
  {choices: [{delta: {content: text}}]},
  {choices: [{delta: {}, finish_reason: 'stop'}]},
  {choices: [], usage: {prompt_tokens: 900, completion_tokens: 20}},
];
const toolOk = data => ({ok: true, json: async () => data});
const found = {results: [
  {title: 'Погода в Минске', url: 'https://yandex.by/pogoda/ru/minsk', snippet: 'Сейчас облачно. ' + 'x'.repeat(300)},
  {title: 'Gismeteo', url: 'https://www.gismeteo.by/weather-minsk-4248/', snippet: 'Прогноз'},
]};
const pageData = {url: 'https://yandex.by/pogoda/ru/minsk', title: 'Погода', text: 'Сегодня +17', truncated: true};
const webHandler = (url, body) => toolOk(url.endsWith('/search') ? found : pageData);

function succeed(request) {
  let sent = false;
  request.resolve({ok: true, body: {getReader: () => ({
    async read() {
      if (sent) return {done: true};
      sent = true;
      return {done: false, value: Buffer.from(
        'data: {"choices":[{"delta":{"content":"answer"}}]}\n\ndata: [DONE]\n\n'
      )};
    },
  })}});
}
"""


@unittest.skipUnless(NODE, "Node.js is required to execute the chat UI tests")
class ChatTest(unittest.TestCase):
    def run_chat(self, program):
        result = subprocess.run(
            [NODE, "-e", HARNESS + program, str(CHAT)],
            capture_output=True,
            text=True,
            timeout=5,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_api_key_is_sent_only_as_a_header_and_not_persisted(self):
        self.run_chat(r"""
const storage = new Map();
const chat = createChat(storage);
chat.elements['api-key'].value = 'test-server-key';
setText(chat, 'hello');
submit(chat);
assert.equal(chat.requests[0].headers.Authorization, 'Bearer test-server-key');
assert.ok(!JSON.stringify(chat.requests[0].body).includes('test-server-key'));
assert.ok(!JSON.stringify([...storage]).includes('test-server-key'));
const fresh = createChat(storage);
assert.equal(fresh.elements['api-key'].value, '');
setText(fresh, 'hello');
submit(fresh);
assert.equal(fresh.requests[0].headers.Authorization, undefined);
""")

    def test_image_attachment_follows_the_served_input_modalities(self):
        self.run_chat(r"""
(async () => {
  const served = modalities => ({ok: true, json: async () => ({data: [{input_modalities: modalities}]})});
  for (const [modalities, accepted] of [
    [['text'], false], [['text', 'image', 'pdf'], true],
  ]) {
    const chat = createChat(new Map(), true, () => served(modalities));
    await flush();
    assert.equal(Boolean(chat.elements.attach.hidden), !accepted, `${modalities}`);
    assert.equal(pasteImages(chat, 'pasted').length, Number(accepted));
    assert.equal(selectImages(chat, 'selected').length, Number(accepted));
  }
  // An offline server leaves the modalities unknown and keeps the button.
  const offline = createChat();
  await flush();
  assert.ok(!offline.elements.attach.hidden, 'offline hid the button');
  // A server that needs a key rejects the first request, which also keeps the
  // button. Entering the key asks again with it; a text-only answer hides the
  // button and drops the image attached meanwhile.
  const chat = createChat(new Map(), true, headers => headers.Authorization ? served(['text'])
    : {ok: false, json: async () => ({error: {type: 'authentication_error'}})});
  await flush();
  assert.ok(!chat.elements.attach.hidden, 'unauthorized hid the button');
  selectImages(chat, 'early');
  assert.equal(chat.elements.attachments.children.length, 1, 'early image not attached');
  chat.elements['api-key'].value = 'key';
  chat.elements['api-key'].handlers.change();
  await flush();
  assert.equal(chat.modelRequests.at(-1).Authorization, 'Bearer key');
  assert.ok(chat.elements.attach.hidden, 'text-only model kept the button');
  assert.equal(chat.elements.attachments.children.length, 0, 'early image kept');
})().catch(error => { console.error(error); process.exitCode = 1; });
""")

    def test_restores_saved_chats_and_effort_with_read_only_storage(self):
        self.run_chat(r"""
const saved = JSON.stringify([{id: 'saved', title: 'Saved chat', updated: 1,
  messages: [{role: 'user', content: 'remembered message'}]}]);
for (const writable of [true, false]) {
  const storage = new Map([
    ['splash-chats', saved], ['splash-thinking-effort', 'low'],
  ]);
  const chat = createChat(storage, writable);
  assert.equal(chat.elements.effort.value, 'low');
  assert.equal(chat.elements.recents.children.length, 1);
  chat.elements.recents.children[0].handlers.click();
  setText(chat, 'continue');
  submit(chat);
  assert.equal(chat.requests[0].body.messages[0].content, 'remembered message');
  assert.equal(storage.get('splash-thinking-effort'), 'low');
}
""")

    def test_default_effort_omits_reasoning_effort(self):
        # The server's --default-reasoning-effort, or the template's default,
        # applies until the user picks an effort.
        self.run_chat(r"""
(async () => {
  const chat = createChat();
  assert.equal(chat.elements.effort.value, '');
  setText(chat, 'hello');
  submit(chat);
  assert.ok(!('reasoning_effort' in chat.requests[0].body));
  succeed(chat.requests[0]);
  await flush();
  chat.elements.effort.value = 'low';
  chat.elements.effort.handlers.change();
  setText(chat, 'again');
  submit(chat);
  assert.equal(chat.requests[1].body.reasoning_effort, 'low');
})().catch(error => { console.error(error); process.exitCode = 1; });
""")

    def test_saves_chats_without_crypto_random_uuid(self):
        # Browsers omit crypto.randomUUID outside secure contexts, such as a
        # LAN address over plain HTTP (#142).
        self.run_chat(r"""
(async () => {
  const {webcrypto} = require('node:crypto');
  const storage = new Map();
  const chat = createChat(storage, true, null,
    {getRandomValues: array => webcrypto.getRandomValues(array)});
  for (const prompt of ['first chat', 'second chat']) {
    setText(chat, prompt);
    submit(chat);
    succeed(chat.requests.at(-1));
    await flush();
    chat.elements['new-chat'].handlers.click();
  }
  const saved = JSON.parse(storage.get('splash-chats'));
  assert.deepEqual(saved.map(item => item.title).sort(), ['first chat', 'second chat']);
  assert.equal(new Set(saved.map(item => item.id)).size, 2);
  for (const {id} of saved) assert.match(id, /^[0-9a-f]{32}$/);
})().catch(error => { console.error(error); process.exitCode = 1; });
""")

    def test_saves_chats_that_exceed_the_storage_quota(self):
        # Browser storage throws once the site holds more than a few MB, which
        # one photo's data URL can fill.
        self.run_chat(r"""
(async () => {
  class Storage extends Map {
    set(key, value) {
      let used = value.length;
      for (const [name, stored] of this) if (name !== key) used += stored.length;
      if (used > 10000) throw Object.assign(new Error('Quota exceeded'), {name: 'QuotaExceededError'});
      return super.set(key, value);
    }
  }
  const storage = new Storage();
  const chat = createChat(storage);
  async function send(text, ...images) {
    chat.elements['new-chat'].handlers.click();
    setText(chat, text);
    pasteImages(chat, ...images).forEach(reading => reading.onload());
    await flush();
    submit(chat);
    succeed(chat.requests.at(-1));
    await flush();
    // The next chat must be newer, since saved chats are ordered by time.
    const last = Date.now();
    while (Date.now() === last) await new Promise(resolve => setTimeout(resolve, 1));
  }
  const saved = () => JSON.parse(storage.get('splash-chats'));
  const notSaved = {type: 'text', text: '[Image not saved]'};
  const screenshot = 'S'.repeat(2000);
  await send('screenshot', screenshot);
  await send('huge photo', 'H'.repeat(12000));
  // A photo too large to store even alone does not cost older chats their images.
  assert.deepEqual(saved().map(conversation => conversation.messages[0].content), [
    [{type: 'text', text: 'huge photo'}, notSaved],
    [{type: 'text', text: 'screenshot'}, imagePart(screenshot)],
  ]);
  const [photoA, photoB] = ['A', 'B'].map(letter => letter.repeat(6000));
  await send('old photo', photoA);
  await send('new photo', photoB);
  // The two photos do not fit together, so the older chat loses its image.
  assert.deepEqual(saved().map(conversation => conversation.messages[0].content), [
    [{type: 'text', text: 'new photo'}, imagePart(photoB)],
    [{type: 'text', text: 'old photo'}, notSaved],
    [{type: 'text', text: 'huge photo'}, notSaved],
    [{type: 'text', text: 'screenshot'}, imagePart(screenshot)],
  ]);
  // Chats that do not fit even without images leave out the oldest chats.
  for (const name of ['one', 'two', 'three']) await send(`${name} ${'x'.repeat(3500)}`);
  assert.deepEqual(saved().map(conversation => conversation.title.split(' ')[0]), ['three', 'two']);
  // The open page keeps every chat with its images.
  assert.equal(chat.elements.recents.children.length, 7);
  chat.elements.recents.children[4].handlers.click();
  setText(chat, 'again');
  submit(chat);
  assert.deepEqual(chat.requests.at(-1).body.messages[0].content,
    [{type: 'text', text: 'old photo'}, imagePart(photoA)]);
})().catch(error => { console.error(error); process.exitCode = 1; });
""")

    def test_enter_preserves_composition_and_shift_but_sends_normal_input(self):
        self.run_chat(r"""
for (const [name, overrides, shouldSend] of [
  ['composition', {isComposing: true}, false],
  ['Safari composition end', {isComposing: false, keyCode: 229}, false],
  ['Shift+Enter', {shiftKey: true}, false],
  ['other key', {key: 'a'}, false],
  ['Enter', {}, true],
]) {
  const {elements, requests} = createChat();
  elements.input.value = 'hello';
  let prevented = false;
  elements.input.handlers.keydown({
    key: 'Enter', keyCode: 13, shiftKey: false, isComposing: false,
    ...overrides,
    preventDefault() { prevented = true; },
  });
  assert.equal(requests.length, Number(shouldSend), name);
  assert.equal(prevented, shouldSend, name);
  assert.equal(elements.input.value, shouldSend ? '' : 'hello', name);
  if (shouldSend) {
    assert.equal(requests[0].url, '/v1/chat/completions');
    assert.deepEqual(requests[0].body.messages, [{role: 'user', content: 'hello'}]);
  }
}
""")

    def test_completion_preserves_the_next_draft_and_restores_failed_attachments(self):
        self.run_chat(r"""
(async () => {
  for (const outcome of ['success', 'stop', 'error']) {
    for (const draft of ['empty', 'ready image', 'loading image']) {
      const chat = createChat();
      const {elements, requests} = chat;
      pasteImage(chat, 'original').onload();
      await flush();
      elements.input.value = 'first prompt';
      submit(chat);
      const original = [{type: 'text', text: 'first prompt'}, imagePart('original')];
      assert.deepEqual(requests[0].body.messages, [{role: 'user', content: original}]);
      assert.deepEqual(draftImages(chat), []);
      let reading;
      if (draft !== 'empty') {
        elements.input.value = 'next prompt';
        reading = pasteImage(chat, 'next');
        if (draft === 'ready image') {
          reading.onload();
          await flush();
          assert.deepEqual(draftImages(chat), [imagePart('next').image_url.url]);
        }
      }
      assert.equal(elements.send.disabled, false, 'Stop stays available while an image loads');
      if (outcome === 'success') succeed(requests[0]);
      else if (outcome === 'stop') submit(chat);
      else requests[0].resolve({ok: false, json: async () => ({error: {message: 'Failed'}})});
      await flush();
      assert.equal(elements.attach.disabled, false, `${outcome}: request must finish`);
      if (draft === 'loading image') {
        assert.equal(elements.send.disabled, true, `${outcome}: unfinished next image blocks Send`);
        submit(chat);
        enter(chat);
        assert.equal(requests.length, 1, `${outcome}: cannot submit an unfinished next image`);
        reading.onload();
        await flush();
        assert.equal(requests.length, 1, 'finishing a read must not queue a request');
      }
      const restored = outcome === 'success' ? [] : [imagePart('original')];
      const expectedImages = [...restored, ...(draft === 'empty' ? [] : [imagePart('next')])];
      const expectedText = draft !== 'empty' ? 'next prompt' : outcome === 'success' ? '' : 'first prompt';
      assert.equal(elements.input.value, expectedText, `${outcome}, ${draft}`);
      assert.deepEqual(draftImages(chat), expectedImages.map(part => part.image_url.url), `${outcome}, ${draft}`);
      // The prior serialized request must stay unchanged as the next draft grows.
      assert.deepEqual(requests[0].body.messages, [{role: 'user', content: original}]);
      if (expectedText || expectedImages.length) {
        submit(chat);
        const expected = [...(expectedText ? [{type: 'text', text: expectedText}] : []), ...expectedImages];
        assert.deepEqual(requests[1].body.messages.at(-1), {role: 'user', content: expected});
        // Successful history remains; stopped/failed requests are removed.
        assert.equal(requests[1].body.messages.length, outcome === 'success' ? 3 : 1);
        succeed(requests[1]);
        await flush();
      }
    }
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
""")

    def test_chunks_without_text_neither_render_nor_scroll(self):
        self.run_chat(r"""
(async () => {
  const scrolls = [];
  for (const count of [0, 3]) {
    const chat = createChat();
    const {elements, requests} = chat;
    elements.input.value = 'prompt';
    submit(chat);
    const empty = Array(count).fill({choices: [{delta: {}}]});
    respond(requests[0], [{choices: [{delta: {role: 'assistant', content: ''}}]},
      ...empty, {choices: [{delta: {content: 'answer'}}]}, ...empty]);
    await flush();
    scrolls.push(chat.scrolls.length);
    elements.input.value = 'next';
    submit(chat);
    assert.equal(requests[1].body.messages[1].content, 'answer');
  }
  assert.equal(scrolls[1], scrolls[0], 'chunks without text must not scroll the page');
})().catch(error => { console.error(error); process.exitCode = 1; });
""")

    def test_loading_image_blocks_send_and_enter_until_ready(self):
        self.run_chat(r"""
(async () => {
  for (const text of ['describe this image', '']) {
    const chat = createChat();
    const {elements, requests} = chat;
    setText(chat, text);
    const reading = pasteImage(chat, 'before-submit');
    // Dispatching submit directly also exercises the guard behind the disabled button.
    submit(chat);
    enter(chat);
    assert.equal(requests.length, 0, 'neither submit nor Enter may send partial content');
    assert.equal(elements.attachments.children.length, 1, 'selection is visible immediately');
    assert.deepEqual(draftImages(chat), [], 'unfinished image is a placeholder');
    assert.equal(elements.send.disabled, true, 'Send is disabled while reading');
    assert.equal(elements.input.value, text);
    reading.onload();
    await flush();
    assert.equal(requests.length, 0, 'completion does not automatically submit');
    assert.equal(elements.send.disabled, false, 'image-only requests become sendable too');
    assert.deepEqual(draftImages(chat), [imagePart('before-submit').image_url.url]);
    if (text) submit(chat);
    else enter(chat);
    assert.deepEqual(requests[0].body.messages, [{role: 'user', content: [
      ...(text ? [{type: 'text', text}] : []), imagePart('before-submit'),
    ]}]);
    succeed(requests[0]);
    await flush();
    assert.deepEqual(draftImages(chat), []);
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
""")

    def test_separate_selections_keep_order_when_reads_finish_out_of_order(self):
        self.run_chat(r"""
(async () => {
  const chat = createChat();
  const [first, second] = selectImages(chat, 'first', 'second');
  const [third] = selectImages(chat, 'third');
  assert.equal(chat.elements.attachments.children.length, 3);
  third.onload();
  second.onload();
  await flush();
  assert.deepEqual(draftImages(chat), ['second', 'third'].map(name => imagePart(name).image_url.url));
  assert.equal(chat.elements.send.disabled, true, 'one pending image blocks the whole request');
  submit(chat);
  assert.equal(chat.requests.length, 0);
  first.onload();
  await flush();
  assert.deepEqual(draftImages(chat), ['first', 'second', 'third'].map(name => imagePart(name).image_url.url));
  submit(chat);
  assert.deepEqual(chat.requests[0].body.messages, [{role: 'user', content:
    ['first', 'second', 'third'].map(imagePart),
  }]);
  succeed(chat.requests[0]);
  await flush();
})().catch(error => { console.error(error); process.exitCode = 1; });
""")

    def test_failed_image_remains_removable_and_preserves_other_selected_images(self):
        self.run_chat(r"""
(async () => {
  const chat = createChat();
  setText(chat, 'first prompt');
  submit(chat);
  const [failed, ready] = pasteImages(chat, 'failed', 'ready');
  failed.error = new Error('Image could not be read');
  failed.onerror();
  ready.onload();
  await flush();
  assert.equal(chat.elements.send.disabled, false, 'Stop remains available with a failed next image');
  submit(chat);
  await flush();
  assert.equal(chat.elements.attach.disabled, false, 'request was stopped');
  const failedItem = chat.elements.attachments.children[0];
  assert.equal(chat.elements.attachments.children.length, 2, 'failure remains visible');
  assert.ok(failedItem.children.some(child => child.tagName === 'SPAN' && child.textContent));
  assert.deepEqual(draftImages(chat), [imagePart('ready').image_url.url]);
  setText(chat, 'keep the readable image');
  assert.equal(chat.elements.send.disabled, true, 'failed image must be removed before sending');
  submit(chat);
  enter(chat);
  assert.equal(chat.requests.length, 1);
  removeImage(chat, 0);
  assert.equal(chat.elements.attachments.children.length, 1);
  assert.equal(chat.elements.send.disabled, false);
  submit(chat);
  assert.deepEqual(chat.requests[1].body.messages, [{role: 'user', content: [
    {type: 'text', text: 'keep the readable image'}, imagePart('ready'),
  ]}]);
  succeed(chat.requests[1]);
  await flush();
})().catch(error => { console.error(error); process.exitCode = 1; });
""")

    def test_removing_pending_image_prevents_late_result_from_returning(self):
        self.run_chat(r"""
(async () => {
  for (const outcome of ['load', 'error']) {
    const chat = createChat();
    const reading = pasteImage(chat, 'removed');
    removeImage(chat, 0);
    assert.equal(chat.elements.attachments.children.length, 0);
    assert.equal(chat.elements.send.disabled, true, 'empty composer stays disabled');
    const next = pasteImage(chat, 'next');
    if (outcome === 'load') reading.onload();
    else {
      reading.error = new Error('Late read failure');
      reading.onerror();
    }
    await flush();
    assert.equal(chat.elements.attachments.children.length, 1, 'removed image cannot return');
    assert.deepEqual(draftImages(chat), []);
    assert.equal(chat.elements.send.disabled, true, 'next image is still loading');
    next.onload();
    await flush();
    submit(chat);
    assert.deepEqual(chat.requests[0].body.messages, [{role: 'user', content: [imagePart('next')]}]);
    succeed(chat.requests[0]);
    await flush();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
""")

    def test_switching_chats_ignores_results_from_previous_draft(self):
        self.run_chat(r"""
(async () => {
  for (const navigation of ['new chat', 'recent chat']) {
    for (const outcome of ['load', 'error']) {
      const chat = createChat();
      setText(chat, 'saved conversation');
      submit(chat);
      succeed(chat.requests[0]);
      await flush();
      const reading = pasteImage(chat, 'old-chat');
      if (navigation === 'new chat') chat.elements['new-chat'].handlers.click();
      else chat.elements.recents.children[0].handlers.click();
      assert.equal(chat.elements.attachments.children.length, 0);
      const next = pasteImage(chat, 'new-chat');
      if (outcome === 'load') reading.onload();
      else {
        reading.error = new Error('Old chat read failure');
        reading.onerror();
      }
      await flush();
      assert.equal(chat.elements.attachments.children.length, 1);
      assert.deepEqual(draftImages(chat), []);
      assert.equal(chat.elements.send.disabled, true);
      next.onload();
      await flush();
      enter(chat);
      assert.deepEqual(chat.requests[1].body.messages.at(-1), {role: 'user', content: [imagePart('new-chat')]});
      assert.equal(chat.requests[1].body.messages.length, navigation === 'new chat' ? 1 : 3);
      succeed(chat.requests[1]);
      await flush();
    }
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
""")

    def test_markdown_renders_structure_and_never_passes_model_html(self):
        self.run_chat(r"""
const {renderMarkdown} = createChat().context;
const table = renderMarkdown('## Key Points\n| Aspect | Detail |\n|---|---|\n| **When to use** | create *families* of objects |');
assert.match(table, /<h2>Key Points<\/h2>/);
assert.match(table, /<th>Aspect<\/th><th>Detail<\/th>/);
assert.match(table, /<td><strong>When to use<\/strong><\/td><td>create <em>families<\/em> of objects<\/td>/);
const diagram = '┌─┐\n│ │\n└─┘\n ▼';
const fenced = renderMarkdown('```text\n' + diagram + '\n<b>x</b> **not bold**\n```');
assert.ok(fenced.includes('<pre><code>' + diagram + '\n&lt;b&gt;x&lt;/b&gt; **not bold**</code></pre>'));
assert.match(renderMarkdown('```js\nlet a = 1'), /<pre><code>let a = 1<\/code><\/pre>/);
const nested = renderMarkdown('1. one\n   - a\n   - b\n2. two\n\n- x\n  1. y');
assert.match(nested, /<ol><li><p>one<\/p><ul><li><p>a<\/p><\/li><li><p>b<\/p><\/li><\/ul><\/li><li><p>two<\/p><\/li><\/ol>/);
assert.match(nested, /<ul><li><p>x<\/p><ol><li><p>y<\/p><\/li><\/ol><\/li><\/ul>/);
assert.match(renderMarkdown('> quote\n\n---\n~~gone~~ `a<b` [ok](https://e.com/a?b=1&c=2)'),
  /<blockquote><p>quote<\/p><\/blockquote><hr><p><del>gone<\/del> <code>a&lt;b<\/code> <a href="https:\/\/e.com\/a\?b=1&amp;c=2" target="_blank" rel="noopener noreferrer">ok<\/a><\/p>/);
for (const evil of ['<img src=x onerror=alert(1)>', '[x](javascript:alert(1))', '<script>alert(1)</script>',
    '[x](https://a.b/"onmouseover="alert(1))', '| <img src=x onerror=alert(1)> | b |\n|---|---|\n| <script> | c |',
    '# <svg onload=alert(1)>', '> <iframe src=x>', '- <a href="javascript:alert(1)">x</a>', '`<script>` **<img src=x>**']) {
  const html = renderMarkdown(evil);
  const tags = html.match(/<[^>]*>/g) || [];
  for (const tag of tags) assert.match(tag, /^<\/?(p|h\d|strong|em|del|code|br|hr|ul|ol|li|table|thead|tbody|tr|th|td|blockquote|div|span|pre|button|a)( |>|$)/, evil + ' -> ' + tag);
  assert.ok(!/<(img|script|svg|iframe)/i.test(html), evil);
  assert.ok(!/href="(?!https?:)/.test(html), evil);
  assert.ok(!/ on\w+=/.test(tags.join('')), evil);
}
""")

    def test_math_is_extracted_rendered_by_katex_and_falls_back_to_raw_tex(self):
        self.run_chat(r"""
const chat = createChat();
const {renderMarkdown} = chat.context;
// Without KaTeX the raw TeX (with delimiters) is shown escaped in <code>.
const sample = 'Let \\(x\\) be the cost of the ball.\n\\[\n\\text{bat} = x + 1.00\n\\]\n\\[\n\\boxed{\\$0.05}\n\\]';
const raw = renderMarkdown(sample);
assert.match(raw, /<code class="math-raw">\\\(x\\\)<\/code>/);
assert.match(raw, /<code class="math-raw">\\\[\n\\text\{bat\} = x \+ 1\.00\n\\\]<\/code>/);
assert.ok(raw.includes('\\boxed{\\$0.05}'));
assert.ok(!/[\u0002\u0003]/.test(raw));
const calls = [];
chat.context.katex = {renderToString(tex, options) {
  calls.push([tex, options.displayMode]);
  assert.deepEqual({...options, displayMode: 0}, {displayMode: 0, throwOnError: false, output: 'html', trust: false, strict: 'ignore'});
  return `<span class="K">${tex.replace(/</g, '&lt;')}</span>`;
}};
const html = (text, ...more) => renderMarkdown(text, ...more);
const seen = text => { calls.length = 0; const out = html(text); return [out, calls.map(([tex, d]) => (d ? 'D:' : 'I:') + tex)]; };
let [out, list0] = seen(sample);
assert.deepEqual(list0, ['I:x', 'D:\n\\text{bat} = x + 1.00\n', 'D:\n\\boxed{\\$0.05}\n']);
assert.ok(!out.includes('math-raw'));
assert.ok(out.includes('\\boxed{\\$0.05}'), 'backslashes survive markdown');
assert.deepEqual(seen('a $$x<y$$ b $z$ c')[1], ['D:x<y', 'I:z']);
assert.deepEqual(seen('$$\na\n\nb\n$$')[1], ['D:\na\n\nb\n']);
// Single $ needs hugging content and no digit after the closing $.
assert.deepEqual(seen('costs $5 and $10 or $ x $ or $x $ or $x$1')[1], []);
assert.deepEqual(seen('price $5, but $x+1$ works; \\$a$ b$')[1], ['I:x+1']);
assert.deepEqual(seen('$a\nb$')[1], []);
// Never inside code.
assert.deepEqual(seen('`$x$ \\(y\\)` and ``\\[z\\]``')[1], []);
assert.deepEqual(seen('```\n$$a$$\n\\(b\\)\n```\n~~~\n$c$\n~~~\n  ```\n$d$\n  ```\nafter $e$')[1], ['I:e']);
assert.deepEqual(seen('```\n$$a$$\n\\(b\\)')[1], []);
// Streaming: an unclosed delimiter stays raw text until closed.
for (const open of ['\\(x', '\\[ x', '$$ x', '$x', 'a \\(q\\) b \\[y']) {
  const [partial, list] = seen(open);
  assert.ok(!partial.includes('class="K"') || open.startsWith('a'), open);
  assert.deepEqual(list, open.startsWith('a') ? ['I:q'] : [], open);
}
assert.deepEqual(seen('\\(w')[1].concat(seen('\\(w\\)')[1]), ['I:w']);
// KaTeX results are cached per tex, so every case above uses its own tex.
// Math inside markdown structure, and markdown next to math.
[out] = seen('- item \\(a\\)\n\n| h |\n|---|\n| $b$ |\n\n**bold** \\(c\\) *em*');
assert.equal((out.match(/class="K"/g) || []).length, 3);
assert.match(out, /<li><p>item <span class="K">a<\/span><\/p><\/li>|<li>item <span class="K">a<\/span><\/li>/);
assert.match(out, /<td><span class="K">b<\/span><\/td>/);
assert.match(out, /<strong>bold<\/strong> <span class="K">c<\/span> <em>em<\/em>/);
// A failing KaTeX falls back to raw TeX.
chat.context.katex = {renderToString() { throw new Error('boom'); }};
assert.match(renderMarkdown('see $a<b$'), /<code class="math-raw">\$a&lt;b\$<\/code>/);
""")

    def test_math_placeholders_cannot_be_forged_or_smuggled_into_attributes(self):
        self.run_chat(r"""
const chat = createChat();
const {renderMarkdown} = chat.context;
chat.context.katex = {renderToString: tex => `<b>${tex}</b>`};
// Model text carrying token-shaped text never expands to math.
for (const forged of ['\u0002abc:0\u0003 \\(x\\)', '\u00020:0\u0003 $y$', '\u0002:0\u0003']) {
  const out = renderMarkdown(forged);
  assert.ok(!/[\u0002\u0003]/.test(out));
  assert.ok(!out.includes('<b>abc') && !/<b>(\\?\(x\\?\))<\/b>/.test(out));
}
// TeX that closes attributes or injects tags is only ever passed to KaTeX, never into markup elsewhere.
const out = renderMarkdown('[a](https://e.com/$x$) [b \\(y\\)](https://e.com/) <img src=x onerror=1> \\(<img onerror=1>\\)');
assert.ok(!/href="[^"]*<b>/.test(out), out);
assert.ok(!/<img/.test(out.replace('<b><img onerror=1></b>', '')), out);
assert.ok(!/[\u0002\u0003]/.test(out));
chat.context.katex = undefined;
assert.ok(!/<img/.test(renderMarkdown('\\(<img onerror=2>\\)')));
""")

    def test_stop_keeps_the_partial_reply_and_regenerate_resends_the_same_history(self):
        self.run_chat(r"""
(async () => {
  const storage = new Map();
  const chat = createChat(storage);
  setText(chat, 'hello');
  submit(chat);
  const sent = chat.requests[0];
  let calls = 0;
  let abortRead;
  sent.resolve({ok: true, body: {getReader: () => ({read: async () => {
    if (calls++) return new Promise((_, reject) => { abortRead = reject; });
    return {done: false, value: Buffer.from('data: {"choices":[{"delta":{"content":"part"}}]}\n\n')};
  }})}});
  await flush();
  submit(chat);
  abortRead(Object.assign(new Error('Stopped'), {name: 'AbortError'}));
  await flush();
  const [saved] = JSON.parse(storage.get('splash-chats'));
  assert.deepEqual(saved.messages.map(item => item.role), ['user', 'assistant']);
  assert.equal(saved.messages[1].content, 'part');
  assert.equal(saved.messages[1].stats.stopped, true);
  chat.elements.recents.children[0].handlers.click();
  setText(chat, 'more');
  submit(chat);
  assert.deepEqual(chat.requests[1].body.messages.map(item => item.content), ['hello', 'part', 'more']);
  assert.ok(!('stats' in chat.requests[1].body.messages[1]));
})().catch(error => { console.error(error); process.exitCode = 1; });
""")

    def test_saved_chats_with_stats_and_system_prompt_load(self):
        self.run_chat(r"""
const saved = JSON.stringify([{id: 'a', title: 'Chat', updated: 1, system: 'be brief',
  messages: [{role: 'user', content: 'q'},
    {role: 'assistant', content: 'a', stats: {ttft_ms: 900, prompt: 10, decode_tps: 20}}]}]);
const chat = createChat(new Map([['splash-chats', saved]]));
assert.equal(chat.elements.recents.children.length, 1);
chat.elements.recents.children[0].handlers.click();
assert.equal(chat.elements['s-system'].value, 'be brief');
assert.match(chat.elements.chat.children[1].children[5].textContent, /TTFT 0\.9 s/);
""")

    def test_request_carries_only_non_default_sampling_and_the_active_temperature(self):
        self.run_chat(r"""
(async () => {
  const storage = new Map();
  const chat = createChat(storage);
  const el = chat.elements;
  const ask = async () => {
    setText(chat, 'hi');
    submit(chat);
    const body = chat.requests.at(-1).body;
    succeed(chat.requests.at(-1));
    await flush();
    el['new-chat'].handlers.click();
    return body;
  };
  const sampling = body => Object.fromEntries(Object.entries(body).filter(
    ([key]) => !['messages', 'stream', 'stream_options', 'reasoning_effort'].includes(key)));
  // New install: Баланс, so only the temperature differs from the server defaults.
  assert.equal(el['gear-label'].textContent, 'T 0.7');
  assert.deepEqual(sampling(await ask()), {temperature: 0.7});
  el['preset-precise'].handlers.click();
  assert.equal(el['gear-label'].textContent, 'T 0.3');
  assert.deepEqual(sampling(await ask()), {temperature: 0.3, top_p: 0.9});
  el['preset-creative'].handlers.click();
  assert.deepEqual(sampling(await ask()), {});
  // Number inputs clamp to what the server accepts; empty max tokens / seed are not sent.
  el['s-top_k-n'].value = '99'; el['s-top_k-n'].handlers.input();
  el['s-max_tokens-n'].value = '512'; el['s-max_tokens-n'].handlers.input();
  el['s-seed-n'].value = '0'; el['s-seed-n'].handlers.input();
  el['s-temperature-n'].value = '1.5'; el['s-temperature-n'].handlers.input();
  assert.deepEqual(sampling(await ask()), {temperature: 1.5, top_k: 32, max_tokens: 512, seed: 0});
  // Settings are global and survive a reload; Сброс sends nothing.
  const reloaded = createChat(storage);
  assert.equal(reloaded.elements['gear-label'].textContent, 'T 1.5');
  reloaded.elements['preset-reset'].handlers.click();
  assert.equal(reloaded.elements['gear-label'].textContent, 'T 1');
  setText(reloaded, 'hi');
  submit(reloaded);
  assert.deepEqual(sampling(reloaded.requests[0].body), {});
  succeed(reloaded.requests[0]);
  await flush();
  assert.deepEqual(JSON.parse(storage.get('splash-sampling')),
    {temperature: 1, top_p: 0.95, top_k: 20, max_tokens: null, seed: null});
})().catch(error => { console.error(error); process.exitCode = 1; });
""")

    def test_system_prompt_is_first_per_chat_and_defaults_to_the_global_one(self):
        self.run_chat(r"""
(async () => {
  const storage = new Map();
  const chat = createChat(storage);
  const el = chat.elements;
  const typeSystem = value => { el['s-system'].value = value; el['s-system'].handlers.input(); el['s-system'].handlers.change(); };
  typeSystem('Answer in Russian');
  assert.equal(el['s-system-hint'].hidden, true, 'no hint before the chat has history');
  setText(chat, 'one');
  submit(chat);
  assert.deepEqual(chat.requests[0].body.messages, [
    {role: 'system', content: 'Answer in Russian'}, {role: 'user', content: 'one'}]);
  succeed(chat.requests[0]);
  await flush();
  setText(chat, 'two');
  submit(chat);
  // The history prefix is byte-identical between turns.
  const [first, second] = chat.requests.map(request => JSON.stringify(request.body.messages));
  assert.ok(second.startsWith(first.slice(0, -2)));
  succeed(chat.requests[1]);
  await flush();
  typeSystem('Answer in English');
  assert.equal(el['s-system-hint'].hidden, false, 'hint after a mid-conversation change');
  assert.equal(JSON.parse(storage.get('splash-chats'))[0].system, 'Answer in English');
  // A new chat starts without it unless it was made the default.
  el['new-chat'].handlers.click();
  assert.equal(el['s-system'].value, '');
  setText(chat, 'plain');
  submit(chat);
  assert.deepEqual(chat.requests[2].body.messages, [{role: 'user', content: 'plain'}]);
  succeed(chat.requests[2]);
  await flush();
  el['new-chat'].handlers.click();
  typeSystem('Default');
  el['s-system-default'].handlers.click();
  el['new-chat'].handlers.click();
  assert.equal(el['s-system'].value, 'Default');
  setText(chat, 'again');
  submit(chat);
  assert.deepEqual(chat.requests[3].body.messages[0], {role: 'system', content: 'Default'});
  succeed(chat.requests[3]);
  await flush();
  // The first chat keeps its own prompt when reopened.
  el.recents.children.find(item => item.children[0].textContent === 'one').handlers.click();
  assert.equal(el['s-system'].value, 'Answer in English');
})().catch(error => { console.error(error); process.exitCode = 1; });
""")

    def test_prefill_rate_is_hidden_for_short_uncached_prompts(self):
        self.run_chat(r"""
(async () => {
  const stats = async (prompt, cached) => {
    const chat = createChat();
    setText(chat, 'hi');
    submit(chat);
    const usage = {prompt_tokens: prompt, completion_tokens: 5, prompt_tokens_details: {cached_tokens: cached}};
    const frames = [
      {choices: [{delta: {content: 'x'}}]},
      {choices: [], usage, timings: {prompt_per_second: 123, predicted_per_second: 20}, metrics: {request_latency: {ttft_ms: 900}}},
    ].map(frame => `data: ${JSON.stringify(frame)}\n\n`).join('') + 'data: [DONE]\n\n';
    let sent = false;
    chat.requests[0].resolve({ok: true, body: {getReader: () => ({read: async () =>
      sent ? {done: true} : (sent = true, {done: false, value: Buffer.from(frames)})})}});
    await flush();
    return chat.elements.chat.children.at(-1).children[5].textContent;
  };
  assert.doesNotMatch(await stats(150, 0), /prefill/);
  assert.doesNotMatch(await stats(5000, 4900), /prefill/, 'cached tokens do not count');
  assert.match(await stats(300, 0), /prefill 123 tok\/s/);
  assert.match(await stats(150, 0), /TTFT 0\.9 s.*decode 20 tok\/s/);
})().catch(error => { console.error(error); process.exitCode = 1; });
""")

    def test_panel_shows_idle_setting_and_countdown_and_saves_changes(self):
        self.run_chat(r"""
(async () => {
  let transport = {ready: true, unloaded: false, recovering: false};
  let current = {idle_unload_seconds: 300, unload_in_seconds: 250};
  let fail = false;
  const status = () => ({ok: true, status: 200, json: async () => ({transport})});
  const settings = options => fail ? {ok: false, statusText: 'Bad', json: async () => ({})} : {ok: true, json: async () => {
    if (options.method === 'POST') current = {idle_unload_seconds: options.body ? JSON.parse(options.body).idle_unload_seconds : 0, unload_in_seconds: 400};
    return current;
  }};
  const chat = createChat(new Map(), true, null, undefined, status, settings);
  const el = chat.elements;
  await flush();
  assert.equal(el['idle-select'].value, '300');
  assert.equal(el['model-sub'].textContent, 'Unloads in 250 s');
  assert.equal(chat.settingsRequests[0].method, 'GET');
  // The countdown ticks locally between the 5 s polls.
  chat.tick(50000);
  assert.equal(el['model-sub'].textContent, 'Unloads in 200 s');
  // A value outside the list is still shown, not silently replaced.
  current = {idle_unload_seconds: 45, unload_in_seconds: 45};
  el['api-key'].value = 'k';
  el['api-key'].handlers.change();
  await flush();
  assert.equal(el['idle-select'].value, '45');
  assert.equal(chat.settingsRequests.at(-1).headers.Authorization, 'Bearer k');
  // Changing the select posts the new value with the key and JSON type.
  el['idle-select'].value = '600';
  el['idle-select'].handlers.change();
  await flush();
  const post = chat.settingsRequests.at(-1);
  assert.equal(post.method, 'POST');
  assert.deepEqual(post.body, {idle_unload_seconds: 600});
  assert.equal(post.headers['Content-Type'], 'application/json');
  assert.equal(post.headers.Authorization, 'Bearer k');
  assert.equal(el['idle-select'].value, '600');
  assert.equal(el['model-sub'].textContent, 'Unloads in 400 s');
  // A failed save restores the previous value and says so.
  fail = true;
  el['idle-select'].value = '0';
  el['idle-select'].handlers.change();
  await flush();
  assert.equal(el['idle-select'].value, '600');
  assert.equal(el['model-sub'].textContent, 'Could not save');
  // Nothing to count down while the model is unloaded.
  fail = false;
  transport = {ready: false, unloaded: true, recovering: false};
  el['api-key'].handlers.change();
  await flush();
  assert.equal(el['model-sub'].textContent, '');
})().catch(error => { console.error(error); process.exitCode = 1; });
""")

    def test_cold_start_wait_is_timed_in_the_panel_not_in_the_message(self):
        self.run_chat(r"""
(async () => {
  const status = () => ({ok: true, status: 200, json: async () => ({transport: {ready: false, unloaded: true, recovering: false}})});
  const chat = createChat(new Map(), true, null, undefined, status);
  await flush();
  const text = () => chat.elements['model-text'].textContent;
  setText(chat, 'hello');
  submit(chat);
  const box = chat.elements.chat.children.at(-1);
  chat.tick(12000);
  assert.equal(text(), 'Loading… 12 s');
  const live = () => box.querySelector('.stats');
  assert.ok(!live().textContent, 'no wait line inside the message');
  let calls = 0, release;
  chat.requests[0].resolve({ok: true, body: {getReader: () => ({read: async () => {
    if (calls++) return new Promise(resolve => { release = resolve; });
    return {done: false, value: Buffer.from('data: {"choices":[{"delta":{"content":"a"}}]}\n\n')};
  }})}});
  await flush();
  assert.equal(text(), 'Model loaded');
  // The message's live line counts generation only, from the first token.
  chat.tick(2000);
  assert.match(live().textContent, /^2\.0 s/);
  release({done: true});
})().catch(error => { console.error(error); process.exitCode = 1; });
""")

    def test_settings_and_api_key_live_in_the_model_panel_not_the_composer(self):
        html = CHAT.read_text()
        panel = html[html.index('id="model-panel"') : html.index("</aside>")]
        composer = html[html.index('<form id="form">') : html.index("</form>")]
        popover = html[html.index('id="settings"') : html.index("<script>")]
        for identifier in ("gear", "gear-label", "idle-select", "model-dot"):
            self.assertIn(f'id="{identifier}"', panel)
            self.assertNotIn(f'id="{identifier}"', composer)
        self.assertIn('id="api-key"', popover)
        self.assertNotIn(
            'id="api-key"', html[: html.index("<script>")].replace(popover, "")
        )

    def test_context_bar_shows_the_last_turn_and_colors_by_share(self):
        self.run_chat(r"""
(async () => {
  const withStats = stats => new Map([['splash-chats', JSON.stringify([{id: 'a', title: 'Chat', updated: 1,
    messages: [{role: 'user', content: 'q'}, {role: 'assistant', content: 'a', ...(stats && {stats})}]}])]]);
  const shown = async (stats, contextLength) => {
    const models = () => ({ok: true, json: async () => ({data: [{input_modalities: ['text'], ...(contextLength && {context_length: contextLength})}]})});
    const chat = createChat(withStats(stats), true, models);
    chat.elements.recents.children[0].handlers.click();
    await flush();
    const {hidden, className} = chat.elements.ctx;
    return {hidden, className, text: chat.elements['ctx-text'].textContent, width: chat.elements['ctx-fill'].style.width};
  };
  assert.deepEqual(await shown({prompt: 8192, completion: 1024}, 32768),
    {hidden: false, className: '', text: '9K / 32K', width: '28.125%'});
  assert.equal((await shown({prompt: 24000, completion: 1000}, 32768)).className, 'warn');
  assert.equal((await shown({prompt: 30000, completion: 1000}, 32768)).className, 'crit');
  // Without /v1/models context_length the default window is 32K.
  assert.equal((await shown({prompt: 1024, completion: 0}, null)).text, '1K / 32K');
  assert.equal((await shown({prompt: 1024, completion: 0}, 65536)).text, '1K / 64K');
  // An old chat without usage shows nothing.
  assert.equal((await shown(null, 32768)).hidden, true);
  assert.equal((await shown({ttft_ms: 900}, 32768)).hidden, true);
  // A new reply moves the bar to that turn's usage.
  const chat = createChat();
  setText(chat, 'hi');
  submit(chat);
  assert.equal(chat.elements.ctx.hidden, true);
  const frames = [{choices: [{delta: {content: 'x'}}]}, {choices: [], usage: {prompt_tokens: 24000, completion_tokens: 3000}}]
    .map(frame => `data: ${JSON.stringify(frame)}\n\n`).join('') + 'data: [DONE]\n\n';
  let sent = false;
  chat.requests[0].resolve({ok: true, body: {getReader: () => ({read: async () =>
    sent ? {done: true} : (sent = true, {done: false, value: Buffer.from(frames)})})}});
  await flush();
  assert.equal(chat.elements['ctx-text'].textContent, '26.4K / 32K');
  assert.equal(chat.elements.ctx.className, 'warn');
})().catch(error => { console.error(error); process.exitCode = 1; });
""")

    def test_thinking_label_shows_duration_and_tokens_and_persists(self):
        self.run_chat(r"""
(async () => {
  const storage = new Map();
  const chat = createChat(storage);
  setText(chat, 'hi');
  submit(chat);
  const box = chat.elements.chat.children.at(-1);
  let step = 0, release;
  const frame = value => Buffer.from(`data: ${JSON.stringify(value)}\n\n`);
  const script = [
    () => frame({choices: [{delta: {reasoning_content: 'hmm'}}]}),
    () => { chat.clock.now += 14000; return frame({choices: [{delta: {content: 'answer'}}]}); },
    () => Buffer.from(`data: ${JSON.stringify({choices: [], usage: {prompt_tokens: 10, completion_tokens: 900,
      completion_tokens_details: {reasoning_tokens: 820}}})}\n\ndata: [DONE]\n\n`),
  ];
  chat.requests[0].resolve({ok: true, body: {getReader: () => ({read: async () =>
    step < script.length ? {done: false, value: script[step++]()} : {done: true}})}});
  await flush();
  const label = () => chat.elements.chat.children.at(-1).children[0].querySelector('summary').textContent;
  assert.equal(label(), 'Thinking · 14 s · 820 tokens');
  const saved = JSON.parse(storage.get('splash-chats'))[0].messages[1];
  assert.ok(Math.abs(saved.stats.reasoning_s - 14) < 0.1);
  assert.equal(saved.stats.reasoning, 820);
  // Reloading the page shows the same label.
  const fresh = createChat(storage);
  fresh.elements.recents.children[0].handlers.click();
  assert.equal(fresh.elements.chat.children.at(-1).children[0].querySelector('summary').textContent, 'Thinking · 14 s · 820 tokens');
  // Messages saved before this data existed keep the plain label.
  const old = new Map([['splash-chats', JSON.stringify([{id: 'a', title: 'Old', updated: 1,
    messages: [{role: 'user', content: 'q'}, {role: 'assistant', content: 'a', reasoning_content: 'r'}]}])]]);
  const legacy = createChat(old);
  legacy.elements.recents.children[0].handlers.click();
  assert.equal(legacy.elements.chat.children.at(-1).children[0].querySelector('summary').textContent, 'Thinking');
})().catch(error => { console.error(error); process.exitCode = 1; });
""")

    def test_model_status_indicator_follows_the_status_endpoint(self):
        self.run_chat(r"""
(async () => {
  let transport = {ready: true, unloaded: false, recovering: false};
  let code = 200;
  const answer = () => ({ok: code === 200, status: code, json: async () => ({transport})});
  const chat = createChat(new Map(), true, null, undefined, answer);
  const text = () => chat.elements['model-text'].textContent;
  const dot = () => chat.elements['model-dot'].className;
  await flush();
  assert.equal(text(), 'Model loaded');
  assert.equal(dot(), 'dot ok');
  // Polling asks /status only, and never the completion endpoint.
  assert.equal(chat.requests.length, 0);
  transport = {ready: false, unloaded: true, recovering: false};
  chat.elements['api-key'].value = 'k';
  chat.elements['api-key'].handlers.change();
  await flush();
  assert.equal(chat.statusRequests.at(-1).Authorization, 'Bearer k');
  assert.ok(text().startsWith('Unloaded'));
  // Sending while unloaded shows loading until the first token.
  setText(chat, 'hello');
  submit(chat);
  assert.equal(text(), 'Loading… 0 s');
  assert.equal(dot(), 'dot busy');
  transport = {ready: true, unloaded: false, recovering: false};
  let calls = 0, abortRead;
  chat.requests[0].resolve({ok: true, body: {getReader: () => ({read: async () => {
    if (calls++) return new Promise((_, reject) => { abortRead = reject; });
    return {done: false, value: Buffer.from('data: {"choices":[{"delta":{"content":"a"}}]}\n\n')};
  }})}});
  await flush();
  assert.equal(text(), 'Model loaded');
  submit(chat);
  abortRead(Object.assign(new Error('Stopped'), {name: 'AbortError'}));
  await flush();
  transport = {ready: false, unloaded: false, recovering: true};
  chat.elements['api-key'].handlers.change();
  await flush();
  assert.equal(text(), 'Loading…');
  transport = {ready: false, unloaded: false, recovering: false};
  chat.elements['api-key'].handlers.change();
  await flush();
  assert.equal(dot(), 'dot down');
  code = 401;
  chat.elements['api-key'].handlers.change();
  await flush();
  assert.equal(text(), 'API key required');
  code = 500;
  chat.elements['api-key'].handlers.change();
  await flush();
  assert.equal(text(), 'Unavailable');
})().catch(error => { console.error(error); process.exitCode = 1; });
""")

    def test_streaming_follows_the_bottom_until_the_user_scrolls_away(self):
        self.run_chat(r"""
(async () => {
  const chat = createChat();
  const {view, scrolls} = chat;
  const scroll = y => { view.y = y; chat.windowHandlers.scroll(); };
  const jump = chat.elements.jump;
  setText(chat, 'hello');
  submit(chat);
  let push, calls = 0;
  const delta = text => Buffer.from(`data: {"choices":[{"delta":{"content":"${text}"}}]}\n\n`);
  chat.requests[0].resolve({ok: true, body: {getReader: () => ({read: () => {
    if (!calls++) return Promise.resolve({done: false, value: delta('a')});
    return new Promise(resolve => { push = text => resolve({done: false, value: delta(text)}); });
  }})}});
  await flush();
  view.height += 100;
  push('b');
  await flush();
  assert.equal(scrolls.at(-1), view.height, 'follows at the bottom');
  assert.equal(jump.hidden, true);
  // The user scrolls up: growth no longer moves the view and the jump button appears.
  scroll(view.y - 300);
  assert.equal(jump.hidden, false);
  const before = scrolls.length;
  view.height += 100;
  push('c');
  await flush();
  assert.equal(scrolls.length, before, 'must not fight the user');
  // A small wheel step near the bottom still counts as near the bottom.
  jump.handlers.click();
  assert.equal(jump.hidden, true);
  assert.equal(view.y, view.height - view.inner);
  view.height += 100;
  push('d');
  await flush();
  assert.equal(scrolls.at(-1), view.height, 'resumed after the jump');
  // Reaching the bottom by hand resumes following as well.
  scroll(view.y - 300);
  assert.equal(jump.hidden, false);
  scroll(view.height - view.inner - 10);
  assert.equal(jump.hidden, true);
  view.height += 100;
  push('e');
  await flush();
  assert.equal(scrolls.at(-1), view.height);
})().catch(error => { console.error(error); process.exitCode = 1; });
""")

    def test_web_off_sends_no_tools_and_hint(self):
        self.run_chat(r"""
const chat = createChat();
setText(chat, 'hi');
submit(chat);
assert.ok(!('tools' in chat.requests[0].body));
assert.deepEqual(chat.requests[0].body.messages.map(item => item.role), ['user']);
""")

    def test_web_tool_loop_searches_reads_one_page_and_answers_with_sources(self):
        self.run_chat(r"""
(async () => {
  const storage = new Map();
  const chat = createChat(storage);
  chat.tool = webHandler;
  chat.elements['api-key'].value = 'key';
  chat.elements.web.handlers.click();
  setText(chat, 'погода Минск');
  submit(chat);
  const first = chat.requests[0].body;
  assert.deepEqual(first.tools.map(tool => tool.function.name), ['web_search', 'read_page']);
  assert.equal(first.messages[0].role, 'system');
  assert.match(first.messages[0].content, /web_search/);
  respond(chat.requests[0], callChunks('call_1', 'web_search', {query: 'погода Минск'}));
  await flush();
  assert.deepEqual(chat.toolRequests.map(item => [item.url, item.body]), [['/splash/tools/search', {query: 'погода Минск'}]]);
  assert.equal(chat.toolRequests[0].headers.Authorization, 'Bearer key');
  const second = chat.requests[1].body;
  assert.deepEqual(second.messages.map(item => item.role), ['system', 'user', 'assistant', 'tool']);
  assert.deepEqual(second.messages[2].tool_calls, [{id: 'call_1', type: 'function', function: {name: 'web_search', arguments: '{"query":"погода Минск"}'}}]);
  assert.equal(second.messages[3].tool_call_id, 'call_1');
  const [one, two] = second.messages[3].content.split('\n\n');
  assert.match(one, /^1\. Погода в Минске\nhttps:\/\/yandex\.by\/pogoda\/ru\/minsk\nСейчас облачно\. x+…$/);
  assert.ok(one.split('\n')[2].length <= 150);
  assert.equal(two, '2. Gismeteo\nhttps://www.gismeteo.by/weather-minsk-4248/\nПрогноз');
  assert.ok(!('stats' in second.messages[2]));
  respond(chat.requests[1], callChunks('call_2', 'read_page', {url: 'https://yandex.by/pogoda/ru/minsk'}));
  await flush();
  assert.deepEqual(chat.toolRequests[1].body, {url: 'https://yandex.by/pogoda/ru/minsk', query: 'погода Минск'});
  const third = chat.requests[2].body.messages;
  assert.deepEqual(third.map(item => item.role), ['system', 'user', 'assistant', 'tool', 'assistant', 'tool']);
  assert.match(third[5].content, /^Source: https:\/\/yandex\.by\/pogoda\/ru\/minsk\nTitle: Погода\n\nСегодня \+17\n\n\[Text truncated\]$/);
  assert.equal(chat.requests[2].body.tools.length, 2);
  respond(chat.requests[2], answerChunks('+17. Источник: https://www.gismeteo.by/weather-minsk-4248/'));
  await flush();
  const [saved] = JSON.parse(storage.get('splash-chats'));
  assert.equal(saved.web, true);
  assert.deepEqual(saved.messages.map(item => item.role), ['user', 'assistant', 'tool', 'assistant', 'tool', 'assistant']);
  const final = saved.messages.at(-1);
  assert.deepEqual(final.sources.map(item => item.url), ['https://yandex.by/pogoda/ru/minsk', 'https://www.gismeteo.by/weather-minsk-4248/']);
  assert.ok(final.stats.search_s > 0);
  assert.equal(final.stats.prompt, 900);
  // Tool steps are shown, tool results are not rendered as messages.
  assert.equal(chat.elements.chat.children.length, 4);
  // Later turns keep the tool messages, and reopening keeps the toggle.
  const later = createChat(storage);
  later.elements.recents.children[0].handlers.click();
  setText(later, 'а завтра?');
  submit(later);
  assert.deepEqual(later.requests[0].body.messages.map(item => item.role),
    ['system', 'user', 'assistant', 'tool', 'assistant', 'tool', 'assistant', 'user']);
  assert.ok(later.requests[0].body.tools);
  assert.ok(!('sources' in later.requests[0].body.messages[6]));
  // Regenerate drops the whole reply, tool steps included.
  const regenerate = chat.elements.chat.children.at(-1).children.at(-1).children.at(-1);
  regenerate.handlers.click();
  assert.deepEqual(chat.requests[3].body.messages.map(item => item.role), ['system', 'user']);
})().catch(error => { console.error(error); process.exitCode = 1; });
""")

    def test_web_limits_are_enforced_per_turn_and_tool_errors_reach_the_model(self):
        self.run_chat(r"""
(async () => {
  const chat = createChat();
  chat.tool = (url, body) => body.query === 'boom' ? {ok: false, statusText: 'x', json: async () => ({error: {message: 'DuckDuckGo blocked'}})} : webHandler(url, body);
  chat.elements.web.handlers.click();
  setText(chat, 'q');
  submit(chat);
  const steps = [['web_search', {query: 'a'}], ['web_search', {query: 'boom'}], ['web_search', {query: 'c'}],
                 ['read_page', {url: 'https://yandex.by/pogoda/ru/minsk'}], ['read_page', {url: 'https://www.gismeteo.by/'}]];
  for (const [index, [name, args]] of steps.entries()) {
    respond(chat.requests[index], callChunks(`c${index}`, name, args));
    await flush();
  }
  // 3rd search and 2nd read never reach the server.
  assert.deepEqual(chat.toolRequests.map(item => item.url), ['/splash/tools/search', '/splash/tools/search', '/splash/tools/fetch']);
  const tools = chat.requests[5].body.messages.filter(item => item.role === 'tool').map(item => item.content);
  assert.equal(tools.length, 5);
  assert.equal(tools[1], 'Error: DuckDuckGo blocked');
  assert.match(tools[2], /^Error: search limit \(2\)/);
  assert.match(tools[4], /^Error: only one page can be read per answer/);
  respond(chat.requests[5], answerChunks('done'));
  await flush();
  assert.equal(chat.requests.length, 6);
  // Unknown tools and bad arguments are answered as errors as well.
  setText(chat, 'again');
  submit(chat);
  respond(chat.requests[6], callChunks('x', 'rm_rf', {}));
  await flush();
  assert.match(chat.requests[7].body.messages.at(-1).content, /unknown tool rm_rf/);
})().catch(error => { console.error(error); process.exitCode = 1; });
""")

    def test_web_rounds_are_capped_by_dropping_the_tools(self):
        self.run_chat(r"""
(async () => {
  const chat = createChat();
  chat.tool = webHandler;
  chat.elements.web.handlers.click();
  setText(chat, 'q');
  submit(chat);
  for (let index = 0; index < 5; index++) {
    assert.ok(chat.requests[index].body.tools, `round ${index}`);
    respond(chat.requests[index], callChunks(`c${index}`, 'web_search', {query: 'again'}));
    await flush();
  }
  assert.ok(!('tools' in chat.requests[5].body));
})().catch(error => { console.error(error); process.exitCode = 1; });
""")

    def test_stop_aborts_the_whole_web_loop_and_keeps_finished_steps(self):
        self.run_chat(r"""
(async () => {
  const storage = new Map();
  const chat = createChat(storage);
  chat.elements.web.handlers.click();
  // Stop while the first search hangs: nothing was produced, so the turn is handed back.
  setText(chat, 'first');
  submit(chat);
  respond(chat.requests[0], callChunks('c0', 'web_search', {query: 'a'}));
  await flush();
  assert.equal(chat.toolRequests.length, 1);
  submit(chat);
  await flush();
  assert.equal(chat.requests.length, 1, 'no further model call after stop');
  assert.equal(chat.elements.input.value, 'first');
  assert.equal(storage.get('splash-chats'), undefined);
  // One finished step, then stop during the second search.
  chat.tool = (url, body) => body.query === 'a' ? webHandler(url, body) : 'hang';
  chat.elements.input.value = '';
  setText(chat, 'second');
  submit(chat);
  respond(chat.requests[1], callChunks('c1', 'web_search', {query: 'a'}));
  await flush();
  respond(chat.requests[2], callChunks('c2', 'web_search', {query: 'b'}));
  await flush();
  submit(chat);
  await flush();
  assert.equal(chat.requests.length, 3);
  const [saved] = JSON.parse(storage.get('splash-chats'));
  assert.deepEqual(saved.messages.map(item => item.role), ['user', 'assistant', 'tool', 'assistant']);
  assert.equal(saved.messages[3].stats.stopped, true);
  assert.ok(!('tool_calls' in saved.messages[3]));
  assert.ok(saved.messages[1].tool_calls);
})().catch(error => { console.error(error); process.exitCode = 1; });
""")

    def test_a_failing_model_call_rolls_back_tool_steps(self):
        self.run_chat(r"""
(async () => {
  const chat = createChat();
  chat.tool = webHandler;
  chat.elements.web.handlers.click();
  setText(chat, 'q');
  submit(chat);
  respond(chat.requests[0], callChunks('c0', 'web_search', {query: 'a'}));
  await flush();
  chat.requests[1].resolve({ok: false, statusText: 'bad', json: async () => ({error: {message: 'engine down'}})});
  await flush();
  assert.equal(chat.elements.input.value, 'q');
  setText(chat, 'q2');
  submit(chat);
  assert.deepEqual(chat.requests[2].body.messages.map(item => item.role), ['system', 'user']);
})().catch(error => { console.error(error); process.exitCode = 1; });
""")


if __name__ == "__main__":
    unittest.main()
