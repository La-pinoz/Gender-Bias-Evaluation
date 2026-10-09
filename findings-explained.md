# Does an AI Resume Screener Care About Your Gender?
### What we found when we asked an AI to screen 10,968 identical-but-for-gender résumés

---

## The short version

- **We ran a fair test.** We took real professional bios, made two copies of each that were identical except for the person's gender, and asked an AI model whether to shortlist each one for a job.
- **For five of the six jobs we tested, the AI showed no gender preference that held up under checking.** That includes software engineering, architecture and dentistry, where you might expect it to favour men.
- **For nursing, it clearly favoured women.** Given the *same* nurse bio, the AI was about **3.4 times more inclined** to say "shortlist" when the candidate was a woman. When the decision was a close call, the effect was far stronger.
- **The preference rarely changed the final yes/no answer.** That happened in about **3 out of 100** nurse comparisons. Most bios were obvious yes or obvious no cases, where the preference made no difference.
- **Smaller, cheaper copies of the AI inherited the same preference.** When we trained a small, fast model to imitate the big one, it learned the nursing preference just as strongly.
- **The bias could be trained out of the small model completely, at no loss in accuracy.** The fix was simple: show it both gender versions of each bio with the same score.

---

## 1. The question

Companies increasingly use AI to sort job applications. If the AI treats two equally qualified people differently because one is a man and one is a woman, that is a bias, and it can quietly decide who gets an interview.

We wanted to answer three questions:
1. Does an AI résumé screener treat men and women differently, all else being equal?
2. If a company replaces the big AI with a smaller, cheaper copy, does the copy inherit the bias?
3. Can that bias be removed, and what does it cost?

---

## 2. How we tested it: "twin" résumés

The core idea is simple, and it's what makes the test fair.

Comparing real men's bios with real women's bios wouldn't work. Their experience, wording and backgrounds differ, so any difference in scores could come from those things instead of gender.

So for every bio we made a **twin**: an exact copy where only the gender clues change.

| Original | Twin |
|---|---|
| "**She** completed **her** residency at Albany Medical Center… **Ms.** Priya Sharma…" | "**He** completed **his** residency at Albany Medical Center… **Mr.** Rahul Sharma…" |

- **What changes:** pronouns (she/he, her/his), titles (Ms/Mr), family words (wife/husband), and the first name.
- **What stays the same:** everything else, including experience, degrees, hospitals, wording and surname.

If the AI scores the twins differently, the only possible reason is gender. Researchers call this a **counterfactual** test: "what would have happened if only this one thing were different?"

We were careful that the twins really differed *only* in gender:
- We automatically checked for leftover gender clues.
- We removed bios where the person's real name slipped through.
- We read samples by eye.
- We left words describing the *job* untouched (for example "women's health" in a nurse's bio), because those describe the work, not the person.

---

## 3. The data

- **Source:** *Bias in Bios*, a public research collection of nearly 400,000 real online professional biographies.
- **Six jobs:** three where most workers are men, and three where most are women.

| Job | Share of women in the source data |
|---|---|
| Software engineer | 16% |
| Architect | 24% |
| Dentist | 35% |
| Teacher | 60% |
| Psychologist | 62% |
| **Nurse** | **91%** |

- **Sample:** 914 bios per job per gender, so **10,968 bios** in total.
- **Two questions per bio:** "Should this person be shortlisted for *their own* job?" (the right answer is yes) and "for a *different, random* job?" (the right answer is no). Each question was asked for both twins.
- **Names:** half the candidates got Western names and half got Indian names, so we could also check whether name origin mattered.
- **Total:** more than **64,000 questions** to the AI, counting the extra checks described below.

---

## 4. The AI and how we read its answers

- **The AI:** *Qwen3-4B*, an openly available AI language model with 4 billion parameters, run on free cloud hardware.
- **The question:** "Should this candidate be shortlisted for an interview for the [job] role? Answer with only Yes or No."

We didn't just record "Yes" or "No". We measured **how strongly** the AI leaned towards each answer, because two "No" answers can differ a lot: one is a hard no, another is nearly a yes.

### Why we use "odds" instead of percentages
Most bios were obvious cases, where the AI was something like 99.9999% sure. Percentages squash all of those together near 0% or 100%, which hides differences. So we used **odds**, the same idea as betting odds:
- 1-to-1 odds means a coin flip.
- 10-to-1 means a strong lean towards Yes.
- 1-to-1,000 means a strong lean towards No.

In the technical results this appears as **log-odds**, simply a convenient scale for odds where 0 means 50/50. The only thing you need to know:

> **A gap of G on this scale means one twin's odds of "Yes" are e^G times the other's.**
> A gap of **0.25** ≈ odds ×1.3 · a gap of **1.0** ≈ odds ×2.7 · a gap of **1.22** ≈ odds ×**3.4** · a gap of **3.0** ≈ odds ×20.

---

## 5. First check: is the AI any good at the job?

A biased screener that is also useless wouldn't tell us much, so we first checked that it screens sensibly.

| Measure | Result | What it means |
|---|---|---|
| Says "Yes" for the person's real job | 61% | It shortlists most suitable candidates |
| Says "Yes" for a random wrong job | 3% | It rarely shortlists unsuitable ones |
| **AUC** | **0.946** | See below |

**AUC, in plain words:** pick one suitable candidate and one unsuitable candidate at random. AUC is the chance that the AI rates the suitable one higher.
- 0.5 = guessing
- 1.0 = perfect
- **0.946 = the AI picks correctly about 95 times in 100**

So it is a competent screener, which makes its behaviour worth studying.

---

## 6. The main result: gender gaps, job by job

For each job we measured the **gap** = the AI's lean towards the female twin minus its lean towards the male twin.
- Positive = favours women.
- Negative = favours men.
- Zero = no preference.

| Job (% women) | Gap | Odds effect | Plausible range (95%) | Real effect? |
|---|---|---|---|---|
| Software engineer (16%) | −0.07 | ×0.93 | −0.31 to +0.15 | No, indistinguishable from zero |
| Architect (24%) | −0.10 | ×0.90 | −0.31 to +0.11 | No |
| Dentist (35%) | +0.08 | ×1.08 | −0.13 to +0.27 | No |
| Teacher (60%) | +0.18 | ×1.20 | +0.01 to +0.36 | Tiny; didn't survive our other checks |
| Psychologist (62%) | +0.23 | ×1.26 | +0.02 to +0.45 | Small; didn't survive our other checks |
| **Nurse (91%)** | **+1.22** | **×3.4** | **+0.98 to +1.46** | **Yes, clearly** |
| All jobs together | +0.26 | ×1.3 | +0.15 to +0.36 | Driven almost entirely by nurse |

![Gender gap by job](assets/gap_by_occupation.png)

### What do "plausible range" and "real effect" mean?
- **Plausible range (confidence interval):** our measurement comes from a sample, so it has some wobble. The range shows where the true value most likely lies. We are 95% confident it is inside.
  - **If the range includes zero**, we can't rule out "no preference at all".
  - **If the whole range sits above zero**, as it does for nurse (0.98 to 1.46), the preference is real.
- **Why our ranges are wider than usual:**
  - Each twin pair used one of only 20 first names per gender and origin.
  - Some names might, on their own, make the AI lean a little one way, so we also allowed for "which names happened to be used" as a source of wobble.
  - Statisticians call this a **name-robust** interval. It is more cautious, and it is the one shown above.
- **p-value**, if you see one: roughly, how surprising our result would be *if there were truly no gender preference*. Tiny values (like 0.001) mean "very surprising", so a real effect is likely.
- **Multiple-testing correction (Holm):** testing six jobs gives six chances for a fluke. We used a standard adjustment that raises the bar for each job so we don't get fooled.

---

## 7. Making sure it's really about gender: three extra checks

### Check 1: does just changing the name do anything? (the "placebo" test)
- **What we did:** we swapped each candidate's first name for a *different name of the same gender* (Rahul → Nikhil, Priya → Neha). The gender stays the same, so any change is just "noise" from names.
- **Result:** the average shift was −0.07, with a plausible range of −0.16 to +0.01, which includes zero. **Swapping names alone does nothing meaningful.**
- **How we used it:** we treated the size of this name noise as a **noise floor**, a bar a gender effect must clear before we believe it.
  - Only **nurse** cleared it, by a wide margin: 1.22 against a floor of about 0.25.
  - Psychologist (+0.23) and teacher (+0.18) didn't.

### Check 2: does it depend on how we ask?
We asked the same questions with three differently worded prompts.

| Job | Prompt A | Prompt B | Prompt C |
|---|---|---|---|
| **Nurse** | +1.25 | +1.30 | **+3.01** |
| Psychologist | +0.03 | +0.64 | +0.57 |
| Others | small, and they flip direction depending on wording | | |

- The nursing preference appeared under **every** wording. Under one wording it was much bigger: odds ×20.
- The other small effects came and went with the wording, a sign they aren't solid.

### Check 3: does the name's origin matter?
- Indian names: +0.27
- Western names: +0.25

No difference. The AI treated Indian and Western names alike.

---

## 8. Does the bias actually change who gets shortlisted?

This is the practical question, and the answer is nuanced.

| | Nurse | All jobs |
|---|---|---|
| Comparisons where the twins got **different Yes/No answers** ("flip rate") | **2.7%** | 2.1% |
| Difference in the chance of being shortlisted (percentage points) | +2.2 | +0.3 |
| Among *genuine nurses*, extra shortlisting rate for the female twin | +3.9 points | – |

- **Most of the time, no.** About 3 in 100 nurse comparisons flipped. Most bios were clear yes or clear no cases, and a preference doesn't change an obvious answer.
- **Close calls are where it matters.** About 6% of comparisons were genuinely borderline, the cases where the AI was unsure. For nurses among those, the gender gap was **3.66**, which means the female twin's odds were about **39 times** higher.

In plain words:

> **The AI's gender preference mostly changes how *confident* it is, not what it decides. But for borderline nursing candidates, the candidate's gender can strongly tip the decision, and it tips it against men.**

---

## 9. Why nurses? Is that really "bias"?

A natural reading is that the AI learned from the world. About 91% of nurses in the data, and in most countries, are women. An AI trained on huge amounts of human text absorbs that pattern, and may have learned "nurse ≈ woman", so a woman "looks more like" a nurse to it.

**It is still a bias in the sense that matters for hiring.** The two candidates had *identical* qualifications. Preferring one because their gender matches a stereotype is exactly what fair hiring rules out. A male nurse with the same experience gets a worse score.

Two more observations make the picture more interesting:
- **The bias is one-sided.** If the AI simply copied job stereotypes, it should also favour men for software engineering (84% men) and architecture (76% men). It didn't: those gaps were about zero, even slightly towards women.
- **So the AI isn't a simple "stereotype machine".** The bias shows up where the stereotype is most extreme (nurse, 91% women). We can't tell from our data whether that's because nursing is the most lopsided job we tested or because something is special about "female-coded" jobs. We only had one strongly female job.

---

## 10. Cheap copies inherit the bias

Running a large AI on every application is expensive. A common shortcut is **distillation**: train a much smaller, faster model to imitate the big one's scores. We did this with a model about 60 times smaller (DistilBERT, 66 million parameters).

**How well did the copy imitate the original?**
- **Correlation** measures how closely two sets of scores move together: 1 = perfect, 0 = unrelated. The small model's correlation with the big one was **0.91**, a very close copy. A simpler keyword-based method got 0.84.
- It was also just as good at screening (AUC **0.96**).

**Did it copy the bias?** Yes, completely.

| | Big AI (teacher) | Small copy (student) |
|---|---|---|
| Gender gap, all jobs | +0.24 | +0.27 |
| Gender gap, nurse | +1.11 | +1.23 |

> **Lesson:** making a model cheaper doesn't make it fairer. The bias travels with the knowledge.

---

## 11. Can the bias be removed?

We tried two ways of training the small model to ignore gender.

1. **Consistency penalty:** while training, the model is penalised whenever it scores a bio's two twins differently. A setting called λ ("lambda") controls how strong the penalty is.
2. **Twin training (counterfactual data augmentation, or CDA):** the model is shown *both* twins with the *same* target score, the average of the two. It learns that gender should never change the answer.

| Small model trained with… | Gender gap, all jobs | Gender gap, nurse | Screening accuracy (AUC) |
|---|---|---|---|
| Ordinary training | +0.27 | +1.23 | 0.960 |
| Consistency penalty, moderate (λ = 0.5) | +0.08 (−72%) | +0.36 (−70%) | 0.960 |
| Consistency penalty, strong (λ = 2) | +0.03 (−90%) | +0.11 (−91%) | 0.961 |
| **Twin training (CDA)** | **−0.03 (≈ gone)** | **−0.02 (≈ gone)** | **0.960** |

![Fairness vs. accuracy](assets/fairness_tradeoff.png)

- **The bias can be removed almost entirely,** and screening accuracy didn't drop at all.
- **The simplest method worked best.** Twin training removed about 98% of the nurse gap.
- **The fix is cheap and should be standard practice** when building screening models.

---

## 12. What this study does and doesn't show

**It does show:**
- One widely available AI screener gave a clear, repeatable preference to women for nursing roles. The result held across names, three prompt wordings, and the placebo check.
- It showed no robust gender preference in the other five jobs.
- Small copies of the AI inherited the bias, and simple training changes removed it without losing accuracy.

**It doesn't show:**
- **That "AI" in general is or isn't biased.** We tested *one* model. Other models, especially larger commercial ones, may behave differently.
- **How a real hiring pipeline behaves.** Real systems use different prompts, rank many candidates together, and involve humans.
- **Anything about non-binary candidates.** We only compared men and women.
- **Whether the effect is about nursing or about very female-dominated jobs in general.** We had only one such job.

**Other caveats:**
- **The exact size depends heavily on wording.** The nurse effect ranged from ×3.5 to ×20 across prompts. The *direction* is solid; the exact number isn't.
- **The data are web bios, not real applications.** About a quarter of candidate bios were set aside during cleaning, to keep the twins truly identical.
- **Two pre-set quality targets for the small model were narrowly missed.** It beat the keyword method by 0.08 instead of the hoped-for 0.10, and on an ordinary CPU it took 60 milliseconds per bio instead of 50 (36 milliseconds after compression). Neither affects the bias findings.

---

## 13. Takeaways

1. **"Is the AI biased?" has a more interesting answer than yes or no.** Here the bias was narrow, concentrated in the most stereotyped job, and one-directional.
2. **Averages hide where bias matters.** Overall, the AI rarely changed its decisions. But on borderline nursing candidates, gender made a large difference.
3. **Bias survives compression.** Cheaper models built from biased ones are biased too.
4. **Bias can be fixed cheaply.** Training on gender-swapped twins removed it at no cost in accuracy.
5. **Test fairness with controls.** Without the name-swap placebo and the multiple prompt wordings, we would have reported five "biased" jobs instead of one.

---

## Glossary

| Term | Plain meaning |
|---|---|
| **Counterfactual / twin pair** | Two copies of the same bio that differ only in gender |
| **Odds / log-odds** | How strongly the AI leans to "Yes". A log-odds gap of G multiplies the odds by e^G |
| **Gap** | Lean towards the female twin minus lean towards the male twin |
| **AUC** | Chance the AI ranks a suitable candidate above an unsuitable one (0.5 = guessing, 1 = perfect) |
| **Confidence interval (plausible range)** | The range where the true value most likely lies (95% confidence). If it excludes 0, the effect is real |
| **Name-robust interval** | A more cautious range that also allows for which names happened to be used |
| **p-value** | How surprising a result would be if there were truly no effect. Small = likely real |
| **Holm correction** | An adjustment that stops us being fooled by luck when testing several jobs |
| **Placebo / noise floor** | Swapping to a same-gender name, to measure how much scores move with no gender change at all |
| **Flip rate** | How often the twins get different Yes/No answers |
| **Borderline case** | A bio where the AI was genuinely unsure (between about 1% and 99% "Yes") |
| **Distillation** | Training a small model to imitate a large one |
| **Correlation** | How closely two sets of scores move together (1 = identical pattern) |
| **CDA (twin training)** | Training on both twins with the same target, so gender can't affect the answer |
| **λ (lambda)** | How strongly the consistency penalty pushes the twins to score the same |

---

## About the study
- **Data:** Bias in Bios (public research dataset)
- **AI screener:** Qwen3-4B-Instruct (open model)
- **Small model:** DistilBERT
- **Compute:** run entirely on free Kaggle GPUs
- **Analysis:** every test, threshold and success criterion was written down before the results were seen. The notebooks, code and per-phase result reports are in this repository (see the README).
