# Regulatory and Governance Plan (UK → EU → global)

> This is a planning outline, not legal or regulatory advice. Regulations in this area are
> changing (UK post-market and pre-market reforms, EU AI Act). Confirm the current position with the MHRA,
> a Notified/Approved Body and a regulatory consultant before you rely on any of it.

## 1. Is it a medical device?

Yes, once used clinically. Software that analyses patient images to detect lesions and inform
diagnosis is Software as a Medical Device (SaMD). Research-only use on de-identified data
under an ethics-approved protocol is not placing a device on the market. That is how the
prototype should be used now. The intended-use notice on every output reflects this.

**Draft intended use (to refine):**
> SecondLook is software intended to assist qualified healthcare professionals in the
> retrospective review of recorded lower-GI colonoscopy and colon capsule endoscopy images and
> video, by highlighting regions that may contain colorectal lesions and indicating which were
> not documented in the procedure report. It does not replace review of the full recording
> by a clinician and is not intended for use during the procedure.

Writing it as an offline "second reader" that only *highlights* findings (not CADx, not
real-time) is the lowest-risk first claim. Each later feature (malignancy risk, real time,
autonomous capsule control) widens the intended use and needs its own conformity assessment.

## 2. Likely classification

| Market | Framework | Likely class (detection-only, assistive) |
|---|---|---|
| Great Britain | UK MDR 2002 (as amended), UKCA via an Approved Body | Check against current MHRA rules. The planned reform aligns software classification with EU Rule 11 (≥ IIa) |
| Northern Ireland / EU | EU MDR 2017/745, CE via a Notified Body | Class IIa under Rule 11 (information used for diagnostic decisions); IIb if a wrong output could cause serious deterioration |
| EU AI Act | High-risk AI (medical device that needs a Notified Body) | AI Act obligations sit on top of MDR. Check application dates for Annex I products |
| USA | FDA | Comparable colonoscopy CADe systems were cleared via De Novo / 510(k) (Class II) |

GB has been accepting CE-marked devices for a transition period. Confirm the current
cut-off dates with MHRA, because a CE mark may be the faster route to the GB market.

## 3. What to build now so certification is achievable later

| Requirement | Standard | Where the prototype already helps |
|---|---|---|
| Quality management system | ISO 13485 | Version control, tests, versioned models |
| Software lifecycle | IEC 62304 (likely Class B) | Modular design, unit/integration tests, documented architecture (DESIGN.md) |
| Risk management | ISO 14971 | Hazards: missed lesion (false reassurance), false flag (unneeded repeat), wrong patient/recording. Mitigations: intended-use notice, blind-segment reporting, input hashing |
| Usability | IEC 62366-1 | Explanation for every flag; decision capture; formative studies with trainees and consultants |
| Health software | IEC 82304-1 | Product-level requirements and documentation |
| Clinical evaluation | MDR Annex XIV / MHRA guidance | Retrospective multi-centre study (DESIGN.md Phase 2), then prospective study |
| AI-specific | MHRA "Software and AI as a Medical Device" programme; Good Machine Learning Practice principles (MHRA/FDA/Health Canada); predetermined change control plans | Model sidecar with training data and metrics; audit trail of model versions |
| Cybersecurity | IEC 81001-5-1; NHS DSPT; Cyber Essentials Plus | Offline by default; localhost-only UI; no outbound connections |

The MHRA **AI Airlock** regulatory sandbox is worth applying to: it is aimed at AI medical devices like this one.

## 4. NHS deployment (Scotland first)

* **Clinical safety:** DCB0129 (manufacturer's clinical safety case and hazard log, signed by a Clinical Safety Officer). The health board completes DCB0160. These are NHS England standards, but NHS Scotland boards usually expect the equivalent.
* **Data protection:** UK GDPR / DPA 2018, a DPIA per health board, and Caldicott Guardian sign-off. On-premises processing simplifies this a lot. Patient data never leaves the board.
* **Information security:** the health board's information security assessment; NHS Scotland procurement may use the Scottish Government's Cyber Resilience frameworks. Also prepare the NHS DTAC (Digital Technology Assessment Criteria), which most NHS buyers will ask for.
* **Research route:** NHS Research Scotland / CSO, NHS REC ethics approval and Safe Haven access for retrospective studies (DATASETS.md).
* **Evidence for buyers:** NICE Evidence Standards Framework for digital health technologies; health-economic case (cost of PCCRC compared with the cost of a second read).

## 5. Sequencing

1. **Now:** research use only, on de-identified data. Set up the QMS, risk file and software development plan. Freeze the intended use.
2. **Phase 1–2:** train and validate the detector. Retrospective clinical study under ethics approval. DCB0129 safety case.
3. **First market:** UKCA (and/or CE under EU MDR) for detection-only offline second look.
4. **Later:** CADx (malignancy risk) and capsule-specific claims as separate submissions; FDA once there is UK/EU evidence.
