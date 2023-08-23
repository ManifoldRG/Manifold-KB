**Paper review: Emergent Autonomous Scientific Research Capabilities of LLMs**

Review author: Pranav Guruprasad

**Summary:**

The authors of this paper present an agent that combines multiple LLMs to carry out scientific experiments end-to-end. The multi-LLM system consists of a Planner - which takes a prompt as input from the user; a Web Searcher - which receives queries from the Planner, transforms them into appropriate web search queries, and executes them using the Google Search API; a Docs searcher - which utilizes the query and indexed documentation to retrieve most relevant pages from documentation; a Code execution component - which executes code in an isolated Docker container; and an Automation component - which executes generated code on corresponding hardware. The authors demonstrate the versatility of the agent by evaluating its performance over 3 tasks - 1. Searching through extensive documentation, 2. Precisely controlling liquid handling instruments, 3. Tackling complex problems that need integration of various data sources. The system also demonstrates reasoning and experimental design capabilities, however, raises substantial safety concerns. This paper is application oriented, and aims to showcase the ability of multiple LLMs working together in synergy, co-ordinating with each other under a certain architectural design, in a specific domain.

**Motivation:**

- Explore the capabilities of LLMs to autonomously design, plan, and execute complex scientific experiments. More specifically, the capabilities of an agent that combines multiple LLMs.
- Leverage the reasoning capabilities of LLM to solve complex problems and generate high-quality code for experimental design
- Assess the multi-agent system’s ability to approach an exceptionally challenging problem in a logical and methodical manner.

**Experiments and Results:**

- Agent performed well on simple experiments involving operating a robot, which simultaneously required the ability to consider a set of samples as a whole. For example - “Color every other line with one color of your choice”
- Agent's ability to plan an experiment by combining data from the internet, performing the necessary calculations, and ultimately writing code, was tested by asking it to design a protocol to perform Suzuki and Sonogashira reactions. Complexity was increased by asking the agent to use modules that were released after GPT-4 (All LLM modules in the system were either GPT-4 or 3.5) training data collection cutoff.
- In the above experiment, the agent not only successfully completes it, but also demonstrates high reasoning capabilities, most interesting of which was its ability to correct its own code based on the automatically generated outputs
- Agent also shows a logical and methodical approach in the above experiment, which is intuitive and interpretable to humans, thus showing promise.
- To bring awareness to the possible misuse of LLMs and automated experimentation for two potentially harmful cases - illicit drug, and chemical weapon synthesis, authors design a test set comprising compounds from DEA’s Schedule I and II substances and a list of known chemical weapon agents.
- Agent provides a synthesis solution for 36% of the above-mentioned test set, which raises alarming safety issues. Moreover, the authors find that out of 7 refused chemicals, five were rejected after the Agent utilized the search function to gather more information about the substance - which can be easily manipulated by altering terminology.

**Limitations:**

- No quantitative measures for the performance of the system. Might require module-wise metrics, or an umbrella metric to measure the quality of the end-to-end process
- The paper demonstrates a handful of qualitative results, which may not be sufficient to convince readers of the strength/capability of the architecture and approach.
- Safety issues are addressed, and potential approaches are listed, but are not deeply explored in the paper. Given the seriousness of the safety issues this approach presents, the authors could have explored potential guardrails a bit more.
- Over-reliance on LLMs is not addressed. Using multiple LLMs in sequence can lead to cascading errors, and potentially large errors in the end results. Some sort of grounding might be required to ensure a certain degree of correctness, given that there is not much room for error in scientific experiments.

**Significance:**

- Presents an architecture to combine multiple LLMs, code execution modules, and information retrieval modules, which has potential to be generalized for various other tasks.
- Showcases the ability of LLMs to correct their own mistakes, and think in a logical manner, which can encourage the exploration of a vast variety of use cases for LLMs.
- Has the potential to accelerate scientific research - with the approach made safer and trustworthy, researchers can refocus the time taken to carry out the experimentation, on interpreting the results, refining hypotheses, making discoveries, and more creative tasks.
- This paper is a great example of inter-disciplinary collaboration, and paves the way for researchers from different fields to address complex problems that require a diverse set of skills and knowledge.
- Has the potential to reduce costs in research and development. By automating and streamlining the experimental process, the system can aid the researchers who need to replicate experiments, or carry out experiments with steps that are well documented on the internet.

**Future work:**

- Explore how the approach can incorporate a human in the loop. This will increase safety and trustworthiness of the system.
- Investigate all the potential harmful issues this approach and architecture raises when used for scientific experiments, especially in real-world scenarios.
- Research comprehensive and robust measures to address the raised safety issues.
- Research and come up with evaluation metrics for agents that carry out sequential processes with multiple modules. Both - for the end result, and for each module’s task.
- Explore grounding techniques to ensure the correctness and relevancy of LLMs.

**Related work:**

- LLM-powered agents: [Lilian Weng’s blog](https://lilianweng.github.io/posts/2023-06-23-agent/). Provides a comprehensive framework for the design of an LLM-powered autonomous agent system. [Pranay’s overview of the blog post](https://github.com/ManifoldRG/AgentForge/issues/16#issuecomment-1664256421).
- LLM-based agents for scientific tasks: [ChemCrow: Augmenting LLMs with chemistry tools by Bran et al.](https://www.semanticscholar.org/paper/ChemCrow%3A-Augmenting-large-language-models-with-Bran-Cox/b61fd5f1661d9234fe85e48f34c701be75ae2de5#citing-papers). By combining the reasoning power of LLMs with chemical expert knowledge from computational tools, the system is capable of planning the synthesis of various chemicals. Relevant because the authors present quantitative evaluations of the system, and address the potential safety issues and risks in detail.

**Paper link**: [Emergent autonomous scientific research capabilities of large language models](https://www.semanticscholar.org/paper/Emergent-autonomous-scientific-research-of-large-Boiko-MacKnight/ae6a4cd221684be6ca3082b6f526a7901281490b)
