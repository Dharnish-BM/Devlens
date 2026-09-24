# GDERS Code Review Comment Annotation Guidelines

**Version:** 1.0.0  
**Phase:** Phase 5A — Taxonomy Validation & Gold-Label Dataset  
**Purpose:** Standardize the manual annotation of GitHub Pull Request review comments into validated technical expertise categories.

---

## 1. General Annotation Principles

1. **Focus on the Core Technical Feedback:** Annotate based on the primary software engineering guidance provided by the reviewer, not merely the presence of a keyword.
2. **Conservative Multi-Label Assignment:** Most review comments address one core issue and should receive **a single label**. Assign a second label only when the comment substantially and genuinely addresses two distinct technical domains (e.g., both database schema design and automated unit testing).
3. **Explicit Non-Technical Rejection:** Conversational pleasantries, single-emoji reactions, bot notices, or simple approvals ("LGTM", "Done", "Thanks") must be tagged as `non_technical` rather than forced into an expertise category.
4. **Insufficient Context:** If a comment is too brief, ambiguous, or lacks enough context to determine what is being reviewed (e.g. "Why?", "What about this?"), assign `insufficient_context`.

---

## 2. Decision Tree for Category Assignment

```
                                [Review Comment]
                                        │
                         Is it conversational / trivial?
                               ┌────────┴────────┐
                             (Yes)              (No)
                               │                 │
                        [non_technical]    Does it have sufficient technical context?
                                                ┌────────┴────────┐
                                              (No)               (Yes)
                                               │                  │
                                     [insufficient_context]  Identify Primary Domain:
                                                                  ├─ Architecture & Design (ARCH_DESIGN)
                                                                  ├─ Testing & QA (TESTING_QUALITY)
                                                                  ├─ Performance & Optimization (PERF_OPTIMIZATION)
                                                                  ├─ Security & Privacy (SECURITY_PRIVACY)
                                                                  ├─ Data Management (DATA_MANAGEMENT)
                                                                  ├─ Infrastructure & DevOps (INFRA_DEVOPS)
                                                                  ├─ Frontend / UI / UX (FRONTEND_UI_UX)
                                                                  ├─ Documentation (DOCUMENTATION)
                                                                  ├─ Code Style & Formatting (CODE_STYLE)
                                                                  └─ Bug Fixing & Logic (BUG_LOGIC)
```

---

## 3. Detailed Category Guidelines & Examples

### 1. Architecture & Software Design (`ARCH_DESIGN`)
- **When to assign:** Feedback targeting modularity, class/interface hierarchy, design patterns, separation of concerns, decoupling, dependency management, or API contract structuring.
- **Positive Example:** *"We should decouple this payment processor behind a generic `PaymentGateway` interface rather than referencing Stripe directly in the controller."*
- **Negative Example:** *"Rename `user_id` to `userId`."* *(Belongs in CODE_STYLE)*

### 2. Testing & Quality Assurance (`TESTING_QUALITY`)
- **When to assign:** Requests to add or modify automated tests (unit, integration, regression, mock fixtures, assertion logic, parameterized test cases).
- **Positive Example:** *"Please add a parameterized unit test verifying that negative inputs raise `ValueError`."*
- **Negative Example:** *"Update the GitHub Actions YAML to add Python 3.12."* *(Belongs in INFRA_DEVOPS)*

### 3. Performance & Optimization (`PERF_OPTIMIZATION`)
- **When to assign:** Feedback addressing runtime latency, algorithmic complexity (O(N) vs O(1)), memory overhead, heap allocations, GC pressure, buffering, or async concurrency throughput.
- **Positive Example:** *"Using a `HashSet` here will avoid an O(N^2) nested loop over the candidate list."*
- **Negative Example:** *"Format this loop to 80 character width."* *(Belongs in CODE_STYLE)*

### 4. Security & Privacy (`SECURITY_PRIVACY`)
- **When to assign:** Identification of vulnerabilities, SQL/command injection, XSS, improper authentication/authorization, insecure random generation, credential leakage, or unsafe memory buffers.
- **Positive Example:** *"Sanitize this user input before passing it into the shell command to prevent command injection."*
- **Negative Example:** *"Add a null check so this doesn't crash."* *(Belongs in BUG_LOGIC)*

### 5. Data Management & Storage (`DATA_MANAGEMENT`)
- **When to assign:** Feedback regarding database queries, SQL optimization, ORM relationships (e.g. N+1 queries), table schema migrations, indexing, or serialization formats (JSON/Protobuf).
- **Positive Example:** *"This query causes an N+1 select; use `.preload(:author)` to batch fetch the authors."*
- **Negative Example:** *"Cache this calculation in an in-memory variable."* *(Belongs in PERF_OPTIMIZATION)*

### 6. Infrastructure & DevOps (`INFRA_DEVOPS`)
- **When to assign:** Feedback on Dockerfiles, Kubernetes manifests, Helm charts, CI/CD workflows, build scripts (Gradle, CMake, Makefile), monitoring, and environment provisioning.
- **Positive Example:** *"Add a liveness probe and resource memory limits to this Kubernetes pod spec."*
- **Negative Example:** *"Write a test verifying this helper."* *(Belongs in TESTING_QUALITY)*

### 7. Frontend, UI & Client-Side (`FRONTEND_UI_UX`)
- **When to assign:** Feedback concerning user interface components (React/Flutter/HTML), CSS styling, animations, widget lifecycle, screen reader accessibility (a11y), or DOM manipulation.
- **Positive Example:** *"Wrap this child widget in a `RepaintBoundary` to prevent unnecessary repainting on state changes."*
- **Negative Example:** *"Return JSON status 200 from this backend API."* *(Belongs in ARCH_DESIGN)*

### 8. Documentation & Readability (`DOCUMENTATION`)
- **When to assign:** Requests for docstrings, Javadoc, API reference docs, README explanations, or comments explaining non-obvious algorithmic decisions.
- **Positive Example:** *"Please document what exception this method throws when the socket disconnects in the docstring."*
- **Negative Example:** *"Remove the blank space on line 42."* *(Belongs in CODE_STYLE)*

### 9. Code Style & Formatting (`CODE_STYLE`)
- **When to assign:** Feedback on naming conventions, indentation, dead code removal, linter warnings, import ordering, or language-specific syntactic idioms without structural refactoring.
- **Positive Example:** *"Nit: use snake_case for local variables to adhere to the repository style guide."*
- **Negative Example:** *"Refactor this class to use the Builder pattern."* *(Belongs in ARCH_DESIGN)*

### 10. Bug Fixing & Algorithmic Logic (`BUG_LOGIC`)
- **When to assign:** Identification of logical bugs, null pointer exceptions, unhandled error cases, off-by-one errors, infinite loops, or incorrect state mutations.
- **Positive Example:** *"If `response` is nil here, accessing `response.body` will raise a nil dereference panic. Add a nil check."*
- **Negative Example:** *"Consider writing a docstring explaining what this method does."* *(Belongs in DOCUMENTATION)*

---

## 4. Multi-Label Guidelines

Assign **two labels** if and only if both conditions are met:
1. The comment makes concrete, actionable requests across two distinct technical categories.
2. Neither category subsumes the other.

*Example of Valid Multi-Label:*
> *"This SQL query does an unindexed table scan (`DATA_MANAGEMENT`), and we also need a regression test verifying the timeout behavior (`TESTING_QUALITY`)."*
> → Labels: `["DATA_MANAGEMENT", "TESTING_QUALITY"]`

*Example of Single-Label (Avoid over-labeling):*
> *"Please add a test verifying the API controller returns 404."*
> → Primary focus is the test. Label: `["TESTING_QUALITY"]` (Do not add `ARCH_DESIGN`).

---

## 5. Non-Technical / Ambiguous Status

- `non_technical`: "LGTM!", "Thanks for reviewing!", "Done 👍", "Acknowledged", "Will fix in next PR."
- `insufficient_context`: "Why?", "Are you sure?", "What about this one?", "See above."
