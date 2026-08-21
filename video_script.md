# CA2 Video Script — Automated Exoplanet Detection

Read this straight through. Stage directions are in **[brackets]** — don't read those out.

Record in five separate takes (A to E) and stitch them together afterwards.

---

## A — Slides

**[Canva in presentation mode, fullscreen. Slide 1.]**

Hello, I'm Prarthana Voosala, student number 201946897. This is my COMP702 MSc project — Automated Exoplanet Detection from Space Telescope Data Using Machine Learning — supervised by Stuart Thomason.

Before I start, a quick word on ethics. This project falls under Data Category A0, which is publicly available, open-access data. Everything I used comes from NASA's Kepler mission through the NASA Exoplanet Archive. There are no human participants involved, and no personal or sensitive data of any kind. All the work followed the University of Liverpool's ethical guidance.

**[Slide 2.]**

So, what's the actual problem here?

NASA's Kepler mission was enormously productive — it produced far more candidate signals than astronomers could realistically check by hand. And the catch is that most of those candidates aren't planets at all. They turn out to be eclipsing binary stars, or instrumental artefacts, or light bleeding in from a background star. Sorting the real ones from the false alarms is slow, manual work, and that's exactly the kind of job machine learning should be able to help with.

The dataset I'm working with is the cumulative Kepler Objects of Interest table — 9,564 objects in total. Of those, 4,839 are false positives, 2,747 are confirmed planets, and 1,978 are still listed as candidates.

What I set out to do was compare three machine learning models: Logistic Regression, Random Forest, and a Multi-Layer Perceptron. The important part is that all three run through exactly the same data, the same preprocessing and the same evaluation. That way, if one model does better than another, it's genuinely because of the model — not because I prepared the data differently for it.

**[Slide 3.]**

These were the essential requirements I set out in my specification, and I've met all of them — importing and preprocessing the dataset, implementing all three models, reporting the full set of metrics, producing the visualisations, and investigating feature importance.

From the desirable list, I also built a web interface for demonstrating predictions, and that's what I'll show you first.

**[Slide 5 — the mockup and the built app side by side.]**

On the left is the interface mockup from my CA1 proposal. On the right is what I actually built.

You can see the numbered workflow carried through, the model selector, the metrics panel, the comparison view. The design I proposed back in June is essentially the application I ended up with.

**[Stop recording.]**

---

## B — The application

**[New take. Browser in fullscreen — press F11. App already running. Do not show yourself starting it.]**

So this is the application. Let me load some data into it.

**[Drag `test_dataset_20_percent.csv` onto the upload panel.]**

What I'm loading here is the held-out test set — 1,902 observations that none of the models ever saw during training. And importantly, that split was made at host-star level, so no star appears on both sides. I'll come back to why that matters when I show you the code.

**[Move the slider to observation 52. Model: Random Forest. Click Predict.]**

Let's take a single observation and run it through. Random Forest classifies this one as a planet candidate, and it's very confident about it. Underneath, the app shows the actual label from the dataset — planet candidate — so it's got this one right.

**[Slider to 1478. Predict.]**

Now let's try the other direction, because a system like this has to be good at both. Here the model correctly identifies a false positive. Being able to rule things out is just as useful as finding planets — arguably more useful, given how much telescope time gets spent chasing false alarms.

**[Slider to 183. Random Forest. Predict.]**

Now here's the case I find most interesting, and it's really the whole reason for comparing models.

Random Forest looks at this observation and says false positive.

**[Switch the dropdown to Logistic Regression. Predict again.]**

Same observation, different model — and Logistic Regression says planet candidate. They completely disagree.

The true label here is false positive, so Random Forest is right and Logistic Regression is wrong. And that's the point. If I'd only built one model, I'd have no way of knowing whether the answer I was getting was reliable. Comparing them systematically is what tells you which one you can actually trust.

**[Click the Model Comparison tab. Scroll slowly through it.]**

This tab pulls together the full results. You've got all three models with their metrics, the confusion matrices so you can see what kind of mistakes each one makes, and the feature correlation from the training set.

Random Forest comes out on top, with an F1 score of 0.895 and a ROC-AUC of 0.967.

Everything on this screen is read from the files my notebook produces — none of these numbers are typed in by hand.

**[Stop recording.]**

---

## C — The code

**[New take. VS Code fullscreen, notebook open with all outputs showing. Use the Outline panel to jump between sections rather than scrolling.]**

Let me take you through the parts of the code that actually matter.

**[Jump to Section 4 — defining the target.]**

First, how I defined what I'm predicting. I group confirmed planets and candidates together on one side, and false positives on the other.

That means the model is really learning triage — is this signal worth following up? — rather than "is this definitely a planet". It's a deliberate choice, and it does come with a trade-off: candidates haven't been independently verified yet, so some of those positive labels carry uncertainty. The alternative was to train only on confirmed planets versus false positives, which gives cleaner labels but throws away a lot of data and answers a slightly different question.

**[Jump to Section 5 — leakage removal. Show the code, then scroll to the printed table.]**

This next part is the most important thing in the whole project.

The KOI table comes with columns that were produced *by* the vetting process — the same process that decided the label I'm trying to predict. Things like disposition flags, vetting scores, and the confirmed planet name, which only exists once something has been confirmed.

If I'd left any of those in, the model wouldn't be predicting anything. It would just be reading the answer back to me, and I'd have got a beautiful score that meant absolutely nothing.

So I remove fourteen columns, and for each one I record why it went. That's this table here.

And rather than just trusting my own judgement, I also sweep every remaining column name for vetting-related keywords — things like "disposition", "flag", "vet". That second pass caught five more columns I'd missed on my first go.

**[Jump to Section 6 — the split.]**

The second protection is how I split the data.

A single star can host several Kepler Objects of Interest, and all of them share the same stellar properties — the star's temperature, its radius, its brightness. So if I did a plain random split, observations from the same star could end up on both sides. The model could then recognise the star rather than learning anything about the transit, and my test score would look much better than it deserved.

So instead I use GroupShuffleSplit grouped on the Kepler ID. That keeps every observation belonging to a given star entirely on one side of the split. And I apply the same grouping to the cross-validation folds, so the same problem can't sneak in there either.

**[Jump to Section 8 — the pipelines.]**

Each model is wrapped in a scikit-learn Pipeline. That's not just tidiness — it means the imputation and the scaling get fitted inside each cross-validation fold, rather than once on the whole training set beforehand. If I'd done it beforehand, every validation fold would have been contaminated by statistics drawn from the very rows it was supposed to be testing.

You'll notice only Logistic Regression and the MLP get a scaler. That's because both are sensitive to how large the numbers are, whereas a Random Forest just splits on thresholds one feature at a time, so scaling would make no difference to it.

**[Jump to Sections 12 and 13 — results and feature importance.]**

And these are the final results, with the ROC curves and the feature importance.

One thing worth flagging honestly: the most influential features here are centroid-offset measurements. Those are closely related to how the vetting process identifies false positives in the first place. They're legitimate physical measurements, not vetting decisions — but they do help explain why the scores are as high as they are, and it's a limitation I'll come back to at the end.

**[Stop recording.]**

---

## D — GitHub

**[New take. Your repository page. Scroll the file list, then click Commits. Keep this short.]**

I used GitHub for version control throughout, which is what I said I'd do in my specification. The repository has the notebook, the Streamlit application, the trained model files and all the exported results.

**[Stop recording.]**

---

## E — Results and closing

**[New take. Back to Canva fullscreen, Slide 6.]**

So, the results.

Random Forest came out best — precision of 0.914, recall of 0.876, an F1 score of 0.895, and ROC-AUC of 0.967.

The cross-validation standard deviations were around 0.003, which is small enough that I'm confident the ranking between the models is real and not just noise.

I also ran the entire pipeline a second time from scratch, and got byte-identical models and byte-identical metrics. So the whole thing is genuinely reproducible.

**[Slide 7.]**

Just to pull together what protects those numbers: the leakage columns removed and documented, the split done at host-star level, and the feature selection computed on training data only — never on the test set.

**[Slide 8.]**

And finally, three limitations I want to be upfront about.

The first is that candidates are treated as planets, so what I've built is really a triage model rather than a planet confirmation model.

The second is the one I mentioned earlier — the strongest features are closely related to the vetting process, which does inflate performance to some degree.

And the third is that my MLP is a neural network baseline working on tabular features. It isn't light-curve deep learning in the way that AstroNet or ExoMiner are, so I'd be careful about comparing my numbers directly to theirs.

To sum up: all the essential requirements from my specification were met, plus the web interface from the desirable list, and the whole thing runs in a consistent and reproducible framework.

Thank you for watching.

**[Stop recording.]**

---

## Checklist before you record

- `test_dataset_20_percent.csv` sitting on the Desktop, ready to drag
- Observation numbers written down: **52**, **1478**, **183**
- Streamlit already running, browser in F11 fullscreen
- Notebook open in VS Code with all outputs visible
- Outlook, Teams and WhatsApp closed; notifications muted
- Canva open in presentation mode in a separate window

If the finished video runs over ten minutes, cut Section D first, then trim the pipeline explanation in Section C.
