import numpy as np

# ---- Gazetteers: the closed vocabularies the bot must recognise as slots. ----
ROOMS = ["conference room", "meeting room", "boardroom", "huddle room",
         "training room"]
DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
        "sunday", "today", "tomorrow"]
TIMES = ["9am", "10am", "11am", "noon", "1pm", "2pm", "3pm", "4pm"]


def tokenize(text):
    return text.lower().replace("?", "").replace("!", "").replace(".", "").split()


class MultinomialNaiveBayes:
    """From-scratch bag-of-words intent classifier. Trained by counting how
    often each word appears per intent, scored in log-space with Laplace
    smoothing so words unseen for a class don't zero out the whole product."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, docs, labels):
        self.classes = sorted(set(labels))
        self.vocab = {}
        for d in docs:
            for w in tokenize(d):
                self.vocab.setdefault(w, len(self.vocab))
        V = len(self.vocab)
        N = len(docs)
        # counts[c] = word occurrences in intent c (start at alpha = smoothing).
        counts = {c: np.full(V, self.alpha) for c in self.classes}
        ndoc = {c: 0 for c in self.classes}
        for d, c in zip(docs, labels):
            ndoc[c] += 1
            for w in tokenize(d):
                counts[c][self.vocab[w]] += 1.0
        self.log_prior = {c: np.log(ndoc[c] / N) for c in self.classes}
        # P(word|intent) = count / total, normalised per class, kept in log-space.
        self.log_lik = {c: np.log(counts[c] / counts[c].sum())
                        for c in self.classes}
        return self

    def predict(self, docs):
        out = []
        for d in docs:
            idx = [self.vocab[w] for w in tokenize(d) if w in self.vocab]
            scores = {c: self.log_prior[c] + self.log_lik[c][idx].sum()
                      for c in self.classes}
            out.append(max(scores, key=scores.get))
        return np.array(out)


def extract_slots(text):
    """Slot filling = tag the entities the booking needs. Scan the utterance
    for known room phrases (multi-word), and day / time tokens."""
    t = " " + text.lower() + " "
    toks = tokenize(text)
    room = next((r for r in ROOMS if " " + r + " " in t), None)
    day = next((d for d in DAYS if d in toks), None)
    time = next((tm for tm in TIMES if tm in toks), None)
    return {"room": room, "day": day, "time": time}


class RoomBookingBot:
    """Understands a request = classify its intent + fill its slots, then act."""

    REPLIES = {
        "greeting": "Hello! I can book meeting rooms. What do you need?",
        "thanks":   "You're welcome! Anything else?",
    }

    def fit(self, docs, intents):
        self.clf = MultinomialNaiveBayes().fit(docs, intents)
        return self

    def respond(self, text):
        intent = self.clf.predict([text])[0]
        if intent in self.REPLIES:
            return self.REPLIES[intent]
        s = extract_slots(text)
        room, day, time = s["room"] or "a room", s["day"] or "that day", s["time"]
        when = "%s%s" % (day, " at " + time if time else "")
        verbs = {"book_room": "Booked", "check_availability": "Checked",
                 "cancel": "Cancelled"}
        return "%s the %s for %s." % (verbs[intent], room, when)


def make_data(seed=0, per_intent=90):
    """Synthetic booking corpus. Task intents plant room+day+time slots via
    templates; chit-chat intents plant none. Gold slots are recorded so slot
    recovery can be scored exactly."""
    rng = np.random.RandomState(seed)
    templates = {
        "book_room": ["book the {r} for {d} at {t}",
                      "i want to reserve {r} on {d} {t}",
                      "can i book {r} {d} at {t} please",
                      "schedule {r} for {d} {t}"],
        "check_availability": ["is the {r} free on {d} at {t}",
                               "check availability of {r} for {d} {t}",
                               "is {r} available {d} at {t}"],
        "cancel": ["cancel my booking for {r} on {d} at {t}",
                   "cancel the {r} reservation for {d} {t}",
                   "please cancel {r} {d} {t}"],
        "greeting": ["hi", "hello there", "hey", "good morning", "greetings"],
        "thanks": ["thanks", "thank you", "cheers", "thanks a lot", "much appreciated"],
    }
    docs, intents, gold = [], [], []
    for intent, temps in templates.items():
        for _ in range(per_intent):
            base = rng.choice(temps)
            if "{r}" in base:
                r, d, t = rng.choice(ROOMS), rng.choice(DAYS), rng.choice(TIMES)
                docs.append(base.format(r=r, d=d, t=t))
                gold.append({"room": r, "day": d, "time": t})
            else:
                docs.append(base)
                gold.append({"room": None, "day": None, "time": None})
            intents.append(intent)
    return docs, np.array(intents), gold


if __name__ == "__main__":
    np.random.seed(0)
    docs, intents, gold = make_data(seed=0)

    # Held-out split: fit on 70%, evaluate on unseen 30%.
    idx = np.random.permutation(len(docs))
    split = int(0.7 * len(docs))
    tr, te = idx[:split], idx[split:]
    Xtr, ytr = [docs[i] for i in tr], intents[tr]
    Xte, yte = [docs[i] for i in te], intents[te]

    bot = RoomBookingBot().fit(Xtr, ytr)
    pred = bot.clf.predict(Xte)
    acc = np.mean(pred == yte)

    # Majority-class baseline: always guess the most common training intent.
    vals, cnts = np.unique(ytr, return_counts=True)
    majority = vals[np.argmax(cnts)]
    base_acc = np.mean(yte == majority)

    # Slot filling scored on test utterances that carry slots; random tagger
    # (uniform pick from each gazetteer) is the baseline.
    slotted = [i for i in te if gold[i]["room"] is not None]
    hits = tot = rand = 0
    for i in slotted:
        got, g = extract_slots(docs[i]), gold[i]
        for key, bank in (("room", ROOMS), ("day", DAYS), ("time", TIMES)):
            tot += 1
            hits += got[key] == g[key]
            rand += 1.0 / len(bank)          # expected accuracy of a random guess
    slot_acc, slot_rand = hits / tot, rand / tot

    print("Utterances: %d   Train: %d   Test: %d   Intents: %d"
          % (len(docs), len(tr), len(te), len(bot.clf.classes)))
    print("Vocabulary size: %d" % len(bot.clf.vocab))
    print("-" * 58)
    print("Intent  accuracy : %.4f   (majority baseline %.4f, '%s')"
          % (acc, base_acc, majority))
    print("Slot    accuracy : %.4f   (random  baseline %.4f)"
          % (slot_acc, slot_rand))
    print("-" * 58)
    for q in ["hello there", "book the boardroom for friday at 2pm",
              "is the huddle room free on tomorrow at 10am",
              "please cancel meeting room monday noon", "thanks a lot"]:
        print("  user: %-44s bot: %s" % (q, bot.respond(q)))
    print("-" * 58)
    print("Beats both baselines: %s"
          % bool(acc > base_acc and slot_acc > slot_rand))
