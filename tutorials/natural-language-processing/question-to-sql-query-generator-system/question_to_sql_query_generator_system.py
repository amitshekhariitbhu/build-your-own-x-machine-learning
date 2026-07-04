import numpy as np

# Question -> SQL generator over a fixed `employees` schema.
# Idea: semantic parsing as slot filling. A SQL query here factors into
#   SELECT <clause> FROM employees [WHERE <col> <op> <value>]
# We predict the SELECT clause and the WHERE (col,op) structure with two
# from-scratch multinomial Naive Bayes classifiers over bag-of-words, and
# copy the literal VALUE straight out of the question. Assembling the three
# pieces reconstructs the exact SQL string.

DEPARTMENTS = ["sales", "engineering", "marketing", "hr", "finance"]
CITIES = ["paris", "london", "tokyo", "berlin", "cairo"]

# SELECT clause -> natural-language phrasings that mean it.
SELECT_BANK = [
    ("COUNT(*)",    ["how many employees", "count the employees",
                     "the number of employees", "how many people"]),
    ("AVG(salary)", ["the average salary", "the mean salary", "average pay"]),
    ("AVG(age)",    ["the average age", "the mean age"]),
    ("SUM(salary)", ["the total salary", "the sum of salaries", "total pay"]),
    ("MAX(salary)", ["the highest salary", "the maximum salary", "the top salary"]),
    ("MIN(salary)", ["the lowest salary", "the minimum salary"]),
    ("MAX(age)",    ["the maximum age", "the oldest age"]),
    ("MIN(age)",    ["the minimum age", "the youngest age"]),
    ("name",        ["the names of employees", "list the employees", "which employees"]),
]

# (col, op, phrasings, value-pool tag) for the optional WHERE filter.
FILTER_BANK = [
    ("department", "=", ["in the {v} department", "who work in {v}", "from the {v} team"], "dept"),
    ("city",       "=", ["based in {v}", "located in {v}", "working in {v}"], "city"),
    ("age",        ">", ["older than {v}", "over {v} years old", "aged above {v}"], "age"),
    ("age",        "<", ["younger than {v}", "under {v} years old"], "age"),
    ("salary",     ">", ["earning more than {v}", "paid over {v}", "with salary above {v}"], "sal"),
    ("salary",     "<", ["earning less than {v}", "with salary below {v}"], "sal"),
]
VALUE_POOL = {"dept": DEPARTMENTS, "city": CITIES,
              "age": [25, 30, 35, 40, 45, 50, 55],
              "sal": [30000, 40000, 50000, 60000, 70000, 80000, 90000]}


def build_sql(select_clause, where_label, value):
    # where_label is "NONE" or "<col> <op>"; value is the literal (str).
    sql = "SELECT %s FROM employees" % select_clause
    if where_label != "NONE":
        col, op = where_label.split()
        v = "'%s'" % value if col in ("department", "city") else str(value)
        sql += " WHERE %s %s %s" % (col, op, v)
    return sql


def make_dataset(n, rng):
    questions, sel_labels, where_labels, sqls = [], [], [], []
    for _ in range(n):
        sc, phrases = SELECT_BANK[rng.randint(len(SELECT_BANK))]
        q = phrases[rng.randint(len(phrases))]
        if rng.rand() < 0.85:                         # 85% carry a WHERE filter
            col, op, tmpls, tag = FILTER_BANK[rng.randint(len(FILTER_BANK))]
            val = VALUE_POOL[tag][rng.randint(len(VALUE_POOL[tag]))]
            q = q + " " + tmpls[rng.randint(len(tmpls))].format(v=val)
            where_label = "%s %s" % (col, op)
            value = str(val)
        else:
            where_label, value = "NONE", ""
        questions.append(q)
        sel_labels.append(sc)
        where_labels.append(where_label)
        sqls.append(build_sql(sc, where_label, value))
    return questions, np.array(sel_labels), np.array(where_labels), sqls


def copy_value(tokens, where_label):
    # Extract the literal from the question given the predicted WHERE column.
    if where_label == "NONE":
        return ""
    col = where_label.split()[0]
    if col == "department":
        for t in tokens:
            if t in DEPARTMENTS:
                return t
    elif col == "city":
        for t in tokens:
            if t in CITIES:
                return t
    else:                                             # numeric age / salary
        for t in tokens:
            if t.isdigit():
                return t
    return ""


class MultinomialNB:
    """Bag-of-words Naive Bayes: fit by counting, predict in log-space."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.log_prior = np.zeros(len(self.classes))
        self.log_lik = np.zeros((len(self.classes), X.shape[1]))
        for i, c in enumerate(self.classes):
            Xc = X[y == c]
            self.log_prior[i] = np.log(Xc.shape[0] / X.shape[0])
            counts = Xc.sum(0) + self.alpha           # Laplace smoothing
            self.log_lik[i] = np.log(counts / counts.sum())
        return self

    def predict(self, X):
        scores = self.log_prior[None, :] + X @ self.log_lik.T
        return self.classes[np.argmax(scores, axis=1)]


def build_vocab(questions):
    vocab = {}
    for q in questions:
        for t in q.split():
            vocab.setdefault(t, len(vocab))
    return vocab


def vectorize(questions, vocab):
    X = np.zeros((len(questions), len(vocab)))
    for i, q in enumerate(questions):
        for t in q.split():
            j = vocab.get(t)
            if j is not None:
                X[i, j] += 1.0
    return X


if __name__ == "__main__":
    np.random.seed(0)
    rng = np.random.RandomState(0)

    q_tr, sel_tr, whr_tr, sql_tr = make_dataset(1600, rng)   # training set
    q_te, sel_te, whr_te, sql_te = make_dataset(400, rng)    # held-out set

    vocab = build_vocab(q_tr)
    Xtr, Xte = vectorize(q_tr, vocab), vectorize(q_te, vocab)

    nb_sel = MultinomialNB().fit(Xtr, sel_tr)                # SELECT clause
    nb_whr = MultinomialNB().fit(Xtr, whr_tr)                # WHERE structure
    pred_sel, pred_whr = nb_sel.predict(Xte), nb_whr.predict(Xte)

    # Assemble full SQL: predicted clauses + value copied from the question.
    pred_sql = [build_sql(s, w, copy_value(q.split(), w))
                for q, s, w in zip(q_te, pred_sel, pred_whr)]
    exact = np.mean([p == t for p, t in zip(pred_sql, sql_te)])
    sel_acc = np.mean(pred_sel == sel_te)
    whr_acc = np.mean(pred_whr == whr_te)

    # Majority baseline: always emit the single most common SQL from training.
    uniq, cnt = np.unique(sql_tr, return_counts=True)
    top_sql = uniq[np.argmax(cnt)]
    base_exact = np.mean([top_sql == t for t in sql_te])
    rand_exact = 1.0 / len(uniq)                             # blind-guess level

    print("Train: %d   Test: %d   Vocab: %d   Distinct SQL(train): %d"
          % (len(q_tr), len(q_te), len(vocab), len(uniq)))
    print("-" * 60)
    print("SELECT-clause accuracy: %.4f" % sel_acc)
    print("WHERE-structure accuracy: %.4f" % whr_acc)
    print("Full-SQL exact match : %.4f  (model)" % exact)
    print("Full-SQL exact match : %.4f  (majority-SQL baseline)" % base_exact)
    print("Full-SQL exact match : %.4f  (random-guess level)" % rand_exact)
    print("-" * 60)
    for i in range(4):
        print("Q: %s" % q_te[i])
        print("   -> %s" % pred_sql[i])
    print("-" * 60)
    print("SUCCESS" if exact > 0.9 and exact > base_exact else "FAIL")
