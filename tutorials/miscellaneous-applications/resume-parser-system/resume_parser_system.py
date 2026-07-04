import numpy as np
import re

# Known skill vocabulary grouped by job category (the parser's "gazetteer").
SKILLS = {
    "data-science": ["python", "pandas", "numpy", "statistics", "sql", "tensorflow", "sklearn"],
    "web-dev":      ["javascript", "react", "html", "css", "node", "typescript", "redux"],
    "devops":       ["docker", "kubernetes", "aws", "terraform", "jenkins", "linux", "ansible"],
    "mobile":       ["kotlin", "swift", "android", "ios", "flutter", "java", "xcode"],
}
CATEGORIES = list(SKILLS)
VOCAB = [s for cat in CATEGORIES for s in SKILLS[cat]]          # flat skill list
VIDX = {s: i for i, s in enumerate(VOCAB)}                      # skill -> column

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}")
YEARS_RE = re.compile(r"(\d+)\s+years")


class ResumeParser:
    """From-scratch resume parser: regex field extraction + gazetteer skill
    matching, then a hand-rolled multinomial Naive Bayes to tag job category."""

    def __init__(self):
        self.log_prior = None      # per-class prior
        self.log_lik = None        # class x skill log-likelihood
        self.classes = None

    # ---- extraction: pull structured fields out of raw resume text ----
    def parse(self, text):
        email = EMAIL_RE.search(text)
        phone = PHONE_RE.search(text)
        years = YEARS_RE.search(text)
        low = text.lower()
        # a skill is present if it appears as a whole token in the text
        toks = set(re.findall(r"[a-z0-9]+", low))
        skills = [s for s in VOCAB if s in toks]
        return {
            "name": text.strip().splitlines()[0].strip(),
            "email": email.group(0) if email else "",
            "phone": re.sub(r"\D", "", phone.group(0)) if phone else "",
            "years": int(years.group(1)) if years else 0,
            "skills": skills,
        }

    def _vec(self, skills):
        v = np.zeros(len(VOCAB))
        for s in skills:
            v[VIDX[s]] += 1.0
        return v

    # ---- multinomial Naive Bayes over the extracted skill counts ----
    def fit(self, parsed, labels):
        X = np.array([self._vec(p["skills"]) for p in parsed])
        y = np.array(labels)
        self.classes = CATEGORIES
        self.log_prior = np.zeros(len(CATEGORIES))
        self.log_lik = np.zeros((len(CATEGORIES), len(VOCAB)))
        for c, cat in enumerate(CATEGORIES):
            Xc = X[y == cat]
            self.log_prior[c] = np.log(len(Xc) / len(X))
            counts = Xc.sum(axis=0) + 1.0                       # Laplace smoothing
            self.log_lik[c] = np.log(counts / counts.sum())
        return self

    def predict(self, parsed):
        X = np.array([self._vec(p["skills"]) for p in parsed])
        scores = self.log_prior + X @ self.log_lik.T            # log posterior
        return [CATEGORIES[i] for i in np.argmax(scores, axis=1)]


def make_resume(cat):
    """Synthesize one resume: planted name/email/phone/skills for a category."""
    first = np.random.choice(["Jane", "Amit", "Li", "Omar", "Sara", "Noah", "Ivy", "Raj"])
    last = np.random.choice(["Smith", "Kumar", "Chen", "Ali", "Vega", "Park", "Bose", "Roy"])
    name = f"{first} {last}"
    email = f"{first.lower()}.{last.lower()}@mail.com"
    digits = f"{np.random.randint(200,999)}{np.random.randint(200,999)}{np.random.randint(1000,9999)}"
    fmt = np.random.randint(3)
    phone = [f"({digits[:3]}) {digits[3:6]}-{digits[6:]}",
             f"{digits[:3]}-{digits[3:6]}-{digits[6:]}",
             f"{digits[:3]}.{digits[3:6]}.{digits[6:]}"][fmt]
    years = np.random.randint(1, 15)
    # mostly on-category skills + a little cross-category noise
    own = list(np.random.choice(SKILLS[cat], size=4, replace=False))
    noise_cat = np.random.choice([c for c in CATEGORIES if c != cat])
    noise = list(np.random.choice(SKILLS[noise_cat], size=1))
    skills = own + noise
    np.random.shuffle(skills)
    text = (f"{name}\n"
            f"Email: {email} | Phone: {phone}\n"
            f"Summary: professional with {years} years of experience.\n"
            f"Skills: {', '.join(s.capitalize() for s in skills)}\n")
    truth = {"name": name, "email": email, "phone": re.sub(r"\D", "", phone),
             "years": years, "category": cat}
    return text, truth


if __name__ == "__main__":
    np.random.seed(0)

    # Build a synthetic corpus of resumes with known ground truth.
    texts, truths = [], []
    for _ in range(320):
        cat = CATEGORIES[np.random.randint(len(CATEGORIES))]
        t, g = make_resume(cat)
        texts.append(t); truths.append(g)

    parser = ResumeParser()
    parsed = [parser.parse(t) for t in texts]

    # ---- Signal 1: exact field extraction (email + phone) ----
    email_acc = np.mean([p["email"] == g["email"] for p, g in zip(parsed, truths)])
    phone_acc = np.mean([p["phone"] == g["phone"] for p, g in zip(parsed, truths)])
    # naive baseline: guess the first whitespace token is the email
    base_email = np.mean([t.split()[0] == g["email"] for t, g in zip(texts, truths)])

    # ---- Signal 2: job-category classification on a held-out split ----
    n = len(parsed); idx = np.random.permutation(n); cut = int(0.7 * n)
    tr, te = idx[:cut], idx[cut:]
    labels = [g["category"] for g in truths]
    parser.fit([parsed[i] for i in tr], [labels[i] for i in tr])
    pred = parser.predict([parsed[i] for i in te])
    acc = np.mean([pred[k] == labels[te[k]] for k in range(len(te))])
    vals, cnts = np.unique([labels[i] for i in tr], return_counts=True)
    majority = vals[np.argmax(cnts)]
    base_acc = np.mean([labels[i] == majority for i in te])

    print("Sample resume text:")
    print(texts[0])
    print("Parsed ->", parsed[0])
    print()
    print("--- Signal 1: field extraction (exact match) ---")
    print("Email extraction accuracy :", round(float(email_acc), 3))
    print("Phone extraction accuracy :", round(float(phone_acc), 3))
    print("First-token baseline      :", round(float(base_email), 3))
    print()
    print("--- Signal 2: job-category tagging (held-out) ---")
    print("Naive Bayes accuracy      :", round(float(acc), 3))
    print("Majority-class baseline   :", round(float(base_acc), 3))
    print()
    ok = email_acc > 0.99 and phone_acc > 0.99 and acc > base_acc + 0.3
    print("RESULT:", "PASS - beats both baselines" if ok else "FAIL")
