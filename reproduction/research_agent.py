import json
import logging
import datetime
from copy import deepcopy

import numpy as np
from tqdm import tqdm
from sklearn.cluster import DBSCAN, KMeans
from sentence_transformers import SentenceTransformer

from reproduction import search_utils


def _strip_reasoning_text(text: str) -> str:
    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    return text.strip()


class ResearchAgent:
    def __init__(
        self,
        output_dir,
        research_goal,
        in_datastore,
        n_research_iters,
        n_queries_per_iter,
        n_webpages_per_query,
        relevance_threshold,
        model_name,
        llm_client,
        tokenizer,
        chunking_granularity,
        embedding_model_name=None,
        index_keyphrase_cluster_size=20,
        search_tool="google",
        serper_gl=None,
        serper_hl=None,
        use_cache=True,
        log_llm_outputs=False,
        enable_reasoning=False,
    ):
        self.research_goal = research_goal
        self.current_datastore = in_datastore
        self.validate_datastore(in_datastore)

        self.n_research_iters = n_research_iters
        self.n_queries_per_iter = n_queries_per_iter
        self.n_webpage_per_query = n_webpages_per_query
        self.relevance_threshold = relevance_threshold

        self.model_name = model_name
        self.llm_client = llm_client
        self.tokenizer = tokenizer
        self.chunking_granularity = chunking_granularity
        assert self.model_name in [x.id for x in self.llm_client.models.list()]
        assert self.chunking_granularity < len(self.tokenizer) - 200

        self.datastore_index = None
        self.embedding_model = None
        self.query_encoding_kwargs = {}
        if embedding_model_name:
            self.embedding_model = SentenceTransformer(
                embedding_model_name, trust_remote_code=True
            ).cuda()
            if "stella" in embedding_model_name:
                self.query_encoding_kwargs = {"prompt_name": "s2s_query"}
        self.index_keyphrase_cluster_size = index_keyphrase_cluster_size

        self.search_tool = search_tool
        self.serper_gl = serper_gl
        self.serper_hl = serper_hl
        self.use_cache = use_cache
        self.log_llm_outputs = log_llm_outputs
        self.enable_reasoning = enable_reasoning
        self.extra_body = None
        if not self.enable_reasoning:
            self.extra_body = {"chat_template_kwargs": {"enable_thinking": False}}

        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        log_file = f"{output_dir}/output.log"
        self.log_file = open(log_file, "a")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def _log_llm_output(self, step_name, content):
        if self.log_llm_outputs:
            self.logger.info("LLM_OUTPUT [%s]: %s", step_name, content)

    def _llm_extra_kwargs(self):
        if self.extra_body is None:
            return {}
        return {"extra_body": self.extra_body}

    def validate_datastore(self, datastore):
        assert isinstance(datastore, dict), "Datastore should be a dictionary."
        assert len(datastore) == 2, "Datastore should contain exactly two keys."
        assert [
            x in ["rules", "examples"] for x in datastore.keys()
        ], "Datastore should contain 'rules' and 'examples' keys."

    def summarize_current_datastore(self, is_first_round=False):
        self.logger.info("Summarizing current datastore...")
        if is_first_round:
            summary = ""
            for rule in self.current_datastore["rules"]:
                summary += f"\n\t- {rule['rule']}"
        else:
            assert self.datastore_index is not None
            summary = ""
            for i_cluster, cluster in enumerate(self.datastore_index["indexed_rules"]):
                summary += f"Knowledge Base Section {i_cluster}: {cluster['title']}\n"
                summary += f"\tSummary: {cluster['summary']}\n\n\n"

        self.logger.info("Datastore Summary: %s", summary)
        return summary

    def index_datastore(self):
        def extract_single_rule_keyphrase(rule_text):
            prompt = f"You are an expert in creating domain-specific knowledge bases. Given the domain description and an item in the knowledge base, write one keyphrase from the item. The keyphrase should identify the most salient information (concept or action) that distinguishs the item from the other items in the knowledge base. The information in domain description itself should not be in the keyphrase, because it is shared by all the items in the knowledge base.\n\n##### Domain Description: {self.research_goal}.\n\n##### Item: {rule_text}##### Keyphrase (a single phrase and nothing else):"
            response = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=5000,
                **self._llm_extra_kwargs(),
            )
            keyphrase = _strip_reasoning_text(
                response.choices[0].message.content
            ).lower()
            self._log_llm_output("extract_single_rule_keyphrase", keyphrase)
            return keyphrase

        def merge_rules(rule_text_list):
            while True:
                rules_prompt = f"You are an expert in creating domain-specific knowledge bases. Given a domain description and some items from the knowledge base, combine similar items to make the list more concise. Output a list of json dicts, each dict corresponding to an item after your processing. Each dict must have two fields. The first field is \"original_items\", a list of items you choose to combine, exactly copied from the original items, and \"new_item\" is a string for the processed items. You should not combine items that are dissimilar. For the items you combine, make sure you cover all the information in the new item but do not write very long sentences. Instead, write a few shorter sentences to make the semantics clear.\n\n##### Domain description: {self.research_goal}.\n\n##### Original items:\n{json.dumps(rule_text_list)}\n\n##### Processed items (output json array of dicts and nothing else):"
                response = self.llm_client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": rules_prompt},
                    ],
                    temperature=0,
                    max_tokens=5000,
                    **self._llm_extra_kwargs(),
                )
                self._log_llm_output(
                    "merge_rules", _strip_reasoning_text(response.choices[0].message.content)
                )

                try:
                    merged_rules = json.loads(
                        _strip_reasoning_text(response.choices[0].message.content)
                        .lstrip("```json")
                        .rstrip("```")
                        .strip()
                    )
                    break
                except Exception:
                    self.logger.info(
                        "Merged rule json parsing failed... Generating merged rule again"
                    )
                    self.logger.info(
                        "Model output: %s",
                        _strip_reasoning_text(response.choices[0].message.content),
                    )
            return merged_rules

        def generate_section_summary(rule_text_list):
            rules_prompt = f"You are an expert in creating domain-specific knowledge bases. Given a domain description and some similar items that form a single section, generate a short paragraph to summarize the topic of these items. The summary should serve as a good introduction to this section in the database. You should take the domain description into account and the summary should distinguish the items from the other potential sections under the same domain.\n\n##### Domain description: {self.research_goal}.\n\n##### Section Items:\n{json.dumps(rule_text_list, indent=4)}\n\n##### Section Summary (just output the summary text and nothing else):"
            response = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": rules_prompt},
                ],
                temperature=0,
                max_tokens=5000,
                **self._llm_extra_kwargs(),
            )
            summary = _strip_reasoning_text(response.choices[0].message.content)
            self._log_llm_output("generate_section_summary", summary)
            return summary

        def generate_section_title(rule_text_list, keyphrases_list):
            rules_prompt = f"You are an expert in creating domain-specific knowledge bases. Given a domain description and some similar items that form a single section, as well as their associated keyphrases, generate a title for this section. You should take the domain description into account and the title should distinguish the items from the other potential sections under the same domain.\n\n##### Domain description: {self.research_goal}.\n\n##### Section Items:\n{json.dumps(rule_text_list, indent=4)}\n\n##### Keyphrases:\n{json.dumps(keyphrases_list)}\n\n##### Section Title (just output the title text and nothing else):"
            response = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": rules_prompt},
                ],
                temperature=0,
                max_tokens=5000,
                **self._llm_extra_kwargs(),
            )
            title = _strip_reasoning_text(response.choices[0].message.content)
            self._log_llm_output("generate_section_title", title)
            return title

        def merge_cluster_titles(cluster_titles):
            while True:
                rules_prompt = f"You are an expert in creating domain-specific knowledge bases. Given the domain definition a list of section titles, combine them into a more concise list by merging titles with the same meaning. Output a list of json dicts, each dict corresponding to an item after your processing. Each dict must have two fields. The first field is \"original_titles\", a list of items you choose to combine, exactly copied from the original items, and \"new_title\" is a string for the processed items. You should only combine titles that are similar enough. If the combined title is so general that it is equivalent to the domain description, do not combine.\n\n##### Domain description: {self.research_goal}.\n\n##### Existing titles:\n{json.dumps(cluster_titles)}\n\n##### Combined section titles (in json list format and nothing else):"
                response = self.llm_client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": rules_prompt},
                    ],
                    temperature=0,
                    max_tokens=5000,
                    **self._llm_extra_kwargs(),
                )
                self._log_llm_output(
                    "merge_cluster_titles",
                    _strip_reasoning_text(response.choices[0].message.content),
                )
                try:
                    merged_titles = json.loads(
                        _strip_reasoning_text(response.choices[0].message.content)
                        .lstrip("```json")
                        .rstrip("```")
                        .strip()
                    )
                    break
                except Exception:
                    self.logger.info(
                        "Merged title json parsing failed... Generating merged title again"
                    )
                    self.logger.info(
                        "Model output: %s",
                        _strip_reasoning_text(response.choices[0].message.content),
                    )
            return merged_titles

        self.logger.info("Extracting rule keyphrases...")
        for rule in tqdm(self.current_datastore["rules"]):
            rule["keyphrase"] = extract_single_rule_keyphrase(rule["rule"])

        rule_embeddings = self.embedding_model.encode(
            [x["keyphrase"] for x in self.current_datastore["rules"]],
            **self.query_encoding_kwargs,
        )
        clustering = DBSCAN(eps=0.1, min_samples=2, metric="cosine").fit(
            rule_embeddings
        )
        self.logger.info("DBSCAN clustering results:")
        clusters = {}
        for i, label in enumerate(clustering.labels_):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(self.current_datastore["rules"][i])

        all_rules_merged = []
        for label, cur_cluster_rules in clusters.items():
            print("\n\n\n===================================\n")
            print(f"Cluster {label}:")
            print(f"Keyphrases: {[x['keyphrase'] for x in cur_cluster_rules]}")
            print("Rules before merging:")
            for rule in cur_cluster_rules:
                print(f"\t- {rule['rule']}")
            print("Rules after merging:")

            if label == -1:
                all_rules_merged += cur_cluster_rules
            else:
                orig_rule_to_source = {
                    x["rule"].strip().lower(): x["sources"]
                    if type(x["sources"]) == list
                    else [x["sources"]]
                    for x in cur_cluster_rules
                }
                rule_text_list = [x["rule"] for x in cur_cluster_rules]
                merged_rule_text_list = merge_rules(rule_text_list)
                for rule_merge_dict in merged_rule_text_list:
                    new_rule = {
                        "rule": rule_merge_dict["new_item"],
                        "keyphrase": extract_single_rule_keyphrase(
                            rule_merge_dict["new_item"]
                        ),
                        "sources": [],
                    }
                    try:
                        for rule in rule_merge_dict["original_items"]:
                            new_rule["sources"] += orig_rule_to_source[
                                rule.strip().lower()
                            ]
                    except Exception:
                        pass
                    all_rules_merged.append(new_rule)

                    print(f"\t- {new_rule['rule']}")

        self.current_datastore["rules"] = all_rules_merged

        self.logger.info("KMeans clustering results:")
        rule_embeddings = self.embedding_model.encode(
            [x["keyphrase"] for x in self.current_datastore["rules"]],
            **self.query_encoding_kwargs,
        )
        clustering = KMeans(
            n_clusters=min(len(rule_embeddings), self.index_keyphrase_cluster_size),
            random_state=42,
        ).fit(rule_embeddings)
        clusters = {}
        for i, label in enumerate(clustering.labels_):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(self.current_datastore["rules"][i])
        output_data = {"datastore": self.current_datastore, "indexed_rules": []}
        for label, rules in clusters.items():
            cur_cluster_log_data = {
                "summary": generate_section_summary([x["rule"] for x in rules]),
                "title": generate_section_title(
                    [x["rule"] for x in rules], [x["keyphrase"] for x in rules]
                ),
                "keyphrases": [x["keyphrase"] for x in rules],
                "rules": [x["rule"] for x in rules],
            }
            output_data["indexed_rules"].append(cur_cluster_log_data)
            print("\n\n\n===================================\n")
            print(f"Cluster {label}:")
            print(json.dumps(cur_cluster_log_data, indent=4))

        self.logger.info("Running final cluster merging...")
        title_merge_output = merge_cluster_titles(
            [cluster["title"] for cluster in output_data["indexed_rules"]]
        )
        title_merge_map = {}
        for title_merge_dict in title_merge_output:
            for original_title in title_merge_dict["original_titles"]:
                title_merge_map[original_title.lower().strip()] = title_merge_dict[
                    "new_title"
                ]

        final_cluster_data = {title: {} for title in title_merge_map.values()}
        for cluster in output_data["indexed_rules"]:
            title = cluster["title"].lower().strip()
            if title in title_merge_map:
                title = title_merge_map[title]
                if len(final_cluster_data[title]) == 0:
                    final_cluster_data[title] = {
                        "title": title,
                        "keyphrases": cluster["keyphrases"],
                        "rules": cluster["rules"],
                    }
                else:
                    final_cluster_data[title]["keyphrases"] += cluster["keyphrases"]
                    final_cluster_data[title]["rules"] += cluster["rules"]
            else:
                final_cluster_data[title] = cluster

        for i_cluster, t in enumerate(final_cluster_data):
            if "summary" not in final_cluster_data[t]:
                final_cluster_data[t]["summary"] = generate_section_summary(
                    final_cluster_data[t]["rules"]
                )
            final_cluster_data[t]["keyphrases"] = list(
                set([x.lower().strip() for x in final_cluster_data[t]["keyphrases"]])
            )

            print("\n\n\n===================================\n")
            print(f"Cluster {i_cluster}:")
            print(json.dumps(final_cluster_data[t], indent=4))

        output_data["indexed_rules"] = list(final_cluster_data.values())
        self.datastore_index = output_data

        cur_time = datetime.datetime.now().strftime("%Y%m%d-%H%M")
        with open(f"{self.output_dir}/indexed_rules_{cur_time}.json", "w") as f:
            json.dump(output_data, f, indent=4)

    def generate_search_queries(self, current_datastore_summary):
        self.logger.info("Generating search queries...")
        meta_query = f"You are an expert in creating domain-specific knowledge bases. Given a research goal and a summary of the current knowledge datastore, you write a few querqies to Google for additional knowledge insufficiently covered by the current knowledge datastore.\n\nYour research goal is: {self.research_goal}.\n\nThe current datastore summary: {current_datastore_summary}.\n\nWrite a list of Google queries that would find webpages that expand the coverage of the datastore. The queries should be in the form of a json list of strings, each string being a query. The queries should be relevant to the research goal and aim to cover gaps in the datastore. The queries should be specific. The queries can either directly ask for a specific information or ask for information from specific source types, which increases the likelihood of finding the right webpages. \n\nQueries (in json list format):"
        while True:
            response = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": meta_query},
                ],
                temperature=0,
                max_tokens=5000,
                **self._llm_extra_kwargs(),
            )
            try:
                self._log_llm_output(
                    "generate_search_queries",
                    _strip_reasoning_text(response.choices[0].message.content),
                )
                queries = json.loads(
                    _strip_reasoning_text(response.choices[0].message.content)
                    .lstrip("```json")
                    .rstrip("```")
                    .strip()
                )
                break
            except Exception:
                self.logger.info("Query list json parsing failed... Generating again")
                pass
        return queries

    def search_single_query(self, query, skip_urls, use_cache=True):
        self.logger.info(f"Running search for query: {query}")
        search_result_urls = search_utils.search_urls(
            query,
            self.n_webpage_per_query,
            use_cache=use_cache,
            tool=self.search_tool,
            serper_gl=self.serper_gl,
            serper_hl=self.serper_hl,
        )
        result = []
        if search_result_urls:
            for i, url in enumerate(search_result_urls):
                if url in skip_urls:
                    self.logger.info(
                        f"Processed {url} - already covered by previous search queries"
                    )
                else:
                    scraped_content = search_utils.scrape_and_parse_with_cache(
                        url, use_cache=use_cache
                    )
                    if scraped_content:
                        text_chunks = search_utils.chunk_text(
                            scraped_content["text"],
                            self.tokenizer,
                            self.chunking_granularity,
                        )

                        for j, chunk in enumerate(text_chunks):
                            result.append(
                                {
                                    "id": len(result) + 1,
                                    "content": chunk,
                                    "source": url,
                                    "chunk_index": j,
                                    "total_chunks": len(text_chunks),
                                }
                            )
                        self.logger.info(
                            f"Processed {url} - extracted {len(text_chunks)} chunks"
                        )

        return result

    def generate_rules_from_single_chunk(self, chunk_data):
        webpage_url = chunk_data["source"]
        self.logger.info(
            f"\nGenerating rules from {webpage_url}, chunk {chunk_data['chunk_index']+1}/{chunk_data['total_chunks']}"
        )

        source_rules = []
        rules_prompt = f"""You are an expert in creating domain-specific knowledge bases. Given a research goal and content from Google, you summarize the relevant knowledge in the form of itemized rule. \n\n Based on the following search results generate rules to represent the relevant knowledge.

Generate specific rules that:
1. Are directly extracted or derived from the search results provided. 
2. Relevant to the research goal.
3. Cover different characteristics.
4. Are specific. Include any relevant nuances or edge cases mentioned. 

VERY IMPORTANT: Your response MUST be a valid JSON array containing objects with these exact fields:
- "rule": the text of the rule
- "supporting_text": the exact quote from the passage that supports this rule

For example, your response should look exactly like this:
[
{{
    "supporting_text": "Direct quote from the passage that supports the first rule",
    "rule": "Rule text goes here"
}},
{{
    "supporting_text": "Another direct quote that supports the second rule",
    "rule": "Another rule goes here"
}}
]

Do not include any explanations, markdown formatting, or additional text before or after the JSON array.
\n\n
### Your research goal: 
{self.research_goal}
\n\n
#### Search Result:
{json.dumps({'url': webpage_url, 'webpage_content': chunk_data['content']})}
\n\n
#### Rule (in json array format):"""

        try:
            rule_response = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that only responds with valid JSON as requested. Never include explanations or text outside the JSON structure.",
                    },
                    {"role": "user", "content": rules_prompt},
                ],
            )

            response_content = rule_response.choices[0].message.content
            response_content = _strip_reasoning_text(response_content)
            self._log_llm_output("generate_rules_from_single_chunk", response_content)

            try:
                parsed_content = json.loads(response_content)

                if isinstance(parsed_content, dict) and any(
                    k in parsed_content for k in ["rules", "data", "results", "items"]
                ):
                    for key in ["rules", "data", "results", "items"]:
                        if key in parsed_content and isinstance(
                            parsed_content[key], list
                        ):
                            new_rules = parsed_content[key]
                            break
                    else:
                        for key, value in parsed_content.items():
                            if isinstance(value, list) and len(value) > 0:
                                new_rules = value
                                break
                        else:
                            new_rules = []
                elif isinstance(parsed_content, list):
                    new_rules = parsed_content
                else:
                    new_rules = []

                if not new_rules:
                    self.logger.info(
                        "No rules parsed for %s chunk %s/%s",
                        webpage_url,
                        chunk_data.get("chunk_index", 0) + 1,
                        chunk_data.get("total_chunks", 1),
                    )

                for rule in new_rules:
                    if isinstance(rule, dict) and "rule" in rule:
                        validated_rule = {
                            "rule": rule.get("rule", f""),
                            "sources": [
                                {
                                    "url": webpage_url,
                                    "supporting_text": rule.get(
                                        "supporting_text",
                                        chunk_data.get("content", "")[:100],
                                    ),
                                    "chunk_index": chunk_data.get("chunk_index", 0),
                                    "total_chunks": chunk_data.get(
                                        "total_chunks", 1
                                    ),
                                }
                            ],
                        }
                        source_rules.append(validated_rule)

            except json.JSONDecodeError as e:
                self.logger.debug(f"Error parsing JSON from LLM response: {str(e)}")
                self.logger.debug(
                    "Response content:",
                    response_content[:200] + "..."
                    if len(response_content) > 200
                    else response_content,
                )

                import re

                json_match = re.search(r"\[\s*\{.*\}\s*\]", response_content, re.DOTALL)
                if json_match:
                    try:
                        json_str = json_match.group(0)
                        new_rules = json.loads(json_str)
                        for rule in new_rules:
                            validated_rule = {
                                "rule": rule.get("rule", f""),
                                "sources": [
                                    {
                                        "url": webpage_url,
                                        "supporting_text": rule.get(
                                            "supporting_text",
                                            chunk_data.get("content", "")[:100],
                                        ),
                                        "chunk_index": chunk_data.get(
                                            "chunk_index", 0
                                        ),
                                        "total_chunks": chunk_data.get(
                                            "total_chunks", 1
                                        ),
                                    }
                                ],
                            }
                            source_rules.append(validated_rule)
                        print("Successfully extracted JSON array from response")
                    except Exception as e:
                        print(f"Error extracting JSON array: {str(e)}")
                else:
                    self.logger.info(
                        "No JSON array found in LLM output for %s chunk %s/%s",
                        webpage_url,
                        chunk_data.get("chunk_index", 0) + 1,
                        chunk_data.get("total_chunks", 1),
                    )
                    pass

        except Exception as e:
            print(
                f"Error generating rules for chunk {chunk_data.get('id', 'unknown')}: {str(e)}"
            )

        return source_rules

    def critique_rule_relevance(self, rule_entry):
        prompt = f"You are an expert in creating domain-specific knowledge bases. Given a research goal and a piece of new knowledge you wanted to add to the knowledge datastore, represented as a rule, judge the relevance of the rule. The rule is relevant if it can be added to the knowledge base that answers the research goal. If the rule is only broadly related to the research goal, uninformative to answering the question posed in the research goal, or in the wrong format (e.g., asking for an action when the research is about definition), it is not relevant. Return your answer in a json dict with a single key 'relevance' and the value on a scale from 0 (irrelevant) to 10 (perfectly relevant).\n\n\n##### Research Goal: {self.research_goal}.\n\n\nNew knowledge (represented as a rule): {rule_entry['rule']}\n\n\n##### Is this knowledge relevant enough? Directly write your evaluation in a json dict and do not write anything else:"

        response = self.llm_client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=1,
            n=5,
            max_tokens=5000,
            **self._llm_extra_kwargs(),
        )

        if self.log_llm_outputs:
            raw_scores = [
                _strip_reasoning_text(x.message.content) for x in response.choices
            ]
            self._log_llm_output("critique_rule_relevance", raw_scores)

        def get_rel_score(x):
            return float(
                list(
                    json.loads(x.lstrip("```json").rstrip("```").split("\n\n")[0].strip())
                    .values()
                )[0]
            )

        rel_scores = []
        for x in response.choices:
            try:
                rel_scores.append(
                    get_rel_score(_strip_reasoning_text(x.message.content))
                )
            except Exception:
                continue
        rule_entry["rule_relevance_score"] = np.mean(rel_scores).item() / 10

        return rule_entry

    def run_policy_research(self):
        self.logger.info("Starting ResearchAgent run...")
        self.logger.info("Research Model: %s", self.model_name)

        seen_urls = set()
        for i in range(self.n_research_iters):
            iter_idx = i + 1
            self.logger.info("\n===================================\n")
            self.logger.info("Iteration %d", iter_idx)
            self.logger.info("Research Goal: %s", self.research_goal)

            self.logger.info("Iteration %d | Step: summarize_datastore", iter_idx)
            current_datastore_summary = self.summarize_current_datastore(
                is_first_round=(i == 0)
            )

            self.logger.info("Iteration %d | Step: generate_queries", iter_idx)
            search_queries = self.generate_search_queries(current_datastore_summary)
            search_queries = search_queries[: self.n_queries_per_iter]
            self.logger.info("Generated queries: %s", search_queries)

            for query in search_queries:
                self.logger.info(
                    "Iteration %d | Step: search_and_scrape | Query: %s",
                    iter_idx,
                    query,
                )
                cur_query_search_result_chunks = self.search_single_query(
                    query, seen_urls, use_cache=self.use_cache
                )
                seen_urls.update([x["source"] for x in cur_query_search_result_chunks])

                self.logger.info(
                    "Iteration %d | Step: filter_chunks | Before: %d",
                    iter_idx,
                    len(cur_query_search_result_chunks),
                )
                cur_query_search_result_chunks = [
                    x
                    for x in cur_query_search_result_chunks
                    if len(self.tokenizer.encode(x["content"])) >= 50
                ]
                self.logger.info(
                    "Iteration %d | Step: filter_chunks | After: %d",
                    iter_idx,
                    len(cur_query_search_result_chunks),
                )

                for webpage_chunk in cur_query_search_result_chunks:
                    self.logger.info(
                        "Iteration %d | Step: extract_rules | %s chunk %d/%d",
                        iter_idx,
                        webpage_chunk.get("source"),
                        webpage_chunk.get("chunk_index", 0) + 1,
                        webpage_chunk.get("total_chunks", 1),
                    )
                    cur_chunk_rules = self.generate_rules_from_single_chunk(
                        webpage_chunk
                    )
                    self.logger.info(
                        "Iteration %d | Step: critique_rules | count=%d",
                        iter_idx,
                        len(cur_chunk_rules),
                    )
                    cur_chunk_rules = [
                        self.critique_rule_relevance(x) for x in cur_chunk_rules
                    ]
                    cur_chunk_rules = [
                        x
                        for x in cur_chunk_rules
                        if x["rule_relevance_score"] > self.relevance_threshold
                    ]
                    self.logger.info("Extracted rules: %s", cur_chunk_rules)

                    self.logger.info(
                        "Iteration %d | Step: insert_rules | count=%d",
                        iter_idx,
                        len(cur_chunk_rules),
                    )
                    for new_rule in cur_chunk_rules:
                        self.current_datastore["rules"].append(new_rule)

                    self.logger.info(
                        "Iteration %d | Step: index_datastore", iter_idx
                    )
                    self.index_datastore()

            self.logger.info(
                "Iteration %d | Step: snapshot_datastore", iter_idx
            )
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
            json.dump(
                self.current_datastore,
                open(f"{self.output_dir}/datastore_iter_{i+1}_{timestamp}.json", "w"),
                indent=4,
            )
