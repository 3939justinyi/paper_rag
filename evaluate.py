from anthropic import Anthropic
from ask import answer

claude = Anthropic()

# Each case is a question plus the key facts a correct answer should contain.
# Write 10 to 15 of these yourself, based on what your papers actually say.
test_cases = [
    {
        "question": "What did Waddington originally mean by epigenetics, and how does today's definition differ?",
        "reference": "Waddington in 1942 coined epigenotype and the epigenetic landscape to describe the developmental processes connecting genotype to phenotype and guiding cell fate. The modern molecular definition is heritable changes in gene expression through cell division without altering the DNA sequence, via DNA methylation, histone modifications, and noncoding RNAs. Greally 2018 warns the term is now ambiguous and should be paired with the specific mechanism meant.",
    },
    {
        "question": "What is DNA methylation, where does it occur, and what is its usual effect?",
        "reference": "It is the addition of a methyl group to cytosine, mostly at CpG dinucleotides, by DNA methyltransferases, with DNMT3a and DNMT3b establishing it de novo and DNMT1 maintaining it through mitosis. It generally represses transcription and provides a stable, mitotically heritable cellular memory, and it can respond to environmental signals like diet. From Jaenisch and Bird 2003.",
    },
    {
        "question": "What is the epitranscriptome, and what did Widagdo and Bredy show about it?",
        "reference": "RNA itself carries reversible marks, chiefly m6A, written by Mettl3 and erased by the demethylase FTO. m6A rises in the mouse medial prefrontal cortex after learning, near the stop codons of plasticity genes, and knocking down FTO, which raises m6A, enhances fear memory consolidation. From Widagdo and Bredy 2016.",
    },
    {
        "question": "How does neuronal activity change DNA methylation, and can methylation be removed?",
        "reference": "Yes, it can be removed. Depolarization decreases methylation at the Bdnf promoter, releasing the MeCP2 HDAC Sin3A repressor complex and boosting transcription, from Martinowich 2003. Active demethylation is driven by Gadd45b, an activity induced immediate early gene required for demethylation of BDNF and FGF promoters and for activity dependent adult hippocampal neurogenesis, from Ma 2009.",
    },
    {
        "question": "What is the evidence that epigenetic marks are required for memory?",
        "reference": "Contextual fear conditioning upregulates DNMTs in the hippocampus, and DNMT inhibitors block memory formation. Learning methylates and silences the memory suppressor gene PP1 and demethylates and activates the plasticity gene reelin, from Miller and Sweatt 2007. Santoni 2024 adds that neurons with hyperacetylated histones are preferentially recruited into the memory trace, so boosting histone acetylation enlarges the engram and reducing it prevents allocation.",
    },
    {
        "question": "How does maternal care program stress responses, and does it translate to humans?",
        "reference": "In rats, high licking and grooming reduces methylation at the hippocampal glucocorticoid receptor promoter, increasing NGFI-A binding and GR levels and dampening the HPA stress response. It emerges in the first week, reverses with cross fostering, and is undone by an HDAC inhibitor, from Weaver 2004. McGowan 2009 found the human parallel, where suicide victims with childhood abuse had increased NR3C1 promoter methylation and lower GR in the hippocampus.",
    },
    {
        "question": "What chromatin changes are linked to depression and antidepressant action?",
        "reference": "Chronic social defeat stress drives lasting repressive histone methylation that downregulates hippocampal Bdnf. The antidepressant imipramine reverses this by increasing histone acetylation and downregulating HDAC5, and forcing HDAC5 overexpression blocks recovery, from Tsankova 2006. Because these marks are reversible they are attractive drug targets, raised by Pena and Nestler 2018 and Hwang 2017.",
    },
    {
        "question": "What role does neuroepigenetics play in addiction?",
        "reference": "About half of addiction risk is genetic and the rest involves environmental priming. Drugs of abuse produce long lasting histone and DNA methylation changes throughout the mesolimbic dopamine reward circuit, such as the nucleus accumbens, that drive enduring behavioral changes, including histone acetylation marks like H3K27ac and accumulation of the transcription factor delta FosB. From Walker and Nestler 2018 and Egervari 2017.",
    },
    {
        "question": "What did Nugent 2015 reveal about how the brain becomes male or female?",
        "reference": "The female brain is the default, actively maintained by DNA methylation that represses masculinizing genes in the preoptic area. Perinatal gonadal steroids, mainly estradiol, lower DNMT activity, causing demethylation and de-repression of masculinizing genes. Inhibiting Dnmts, or knocking out Dnmt3a, masculinizes the brain and behavior of females. From Nugent 2015.",
    },
    {
        "question": "What is transgenerational epigenetic inheritance, and what is the landmark behavioral demonstration?",
        "reference": "Environmental information can be transmitted to offspring through the germline, before they are conceived. Dias 2013 fear conditioned F0 male mice to the odor acetophenone, and the F1 and F2 generations inherited heightened sensitivity to that specific odor plus enlarged Olfr151 olfactory circuitry, with CpG hypomethylation at Olfr151 in sperm, transmitted through gametes and confirmed by IVF rather than social learning. Bale 2015 frames this as germline epigenetic reprogramming by environmental stress.",
    },
]


def judge(question, reference, candidate):
    prompt = (
        "You are grading an answer. Given the question, a reference answer containing "
        "the key facts, and a candidate answer, decide whether the candidate captures "
        "the key facts in the reference. Reply with exactly PASS or FAIL on the first "
        "line, then a one sentence reason.\n\n"
        f"Question: {question}\n\n"
        f"Reference answer: {reference}\n\n"
        f"Candidate answer: {candidate}"
    )
    response = claude.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


def run_eval():
    passed = 0
    for i, case in enumerate(test_cases, 1):
        candidate, _ = answer(case["question"])
        verdict = judge(case["question"], case["reference"], candidate)
        if verdict.upper().startswith("PASS"):
            passed += 1
        print(f"{i}. {case['question']}")
        print(f"   {verdict}\n")
    print(f"Score: {passed}/{len(test_cases)} passed")


if __name__ == "__main__":
    run_eval()