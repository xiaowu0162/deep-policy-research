import argparse
import json
import os

from openai import OpenAI
from transformers import AutoTokenizer

from reproduction.research_agent import ResearchAgent

OPENAI_DOMAIN_TO_BASE_RULE = {
    "Sexual": "Content meant to arouse sexual excitement, such as the description of sexual activity, or that promotes sexual services (excluding sex education and wellness).",
    "Hate": "Content that expresses, incites, or promotes hate based on race, gender, ethnicity, religion, nationality, sexual orientation, disability status, or caste.",
    "Violence": "Content that promotes or glorifies violence or celebrates the suffering or humiliation of others.",
    "Harassment": "Content that may be used to torment or annoy individuals in real life, or make harassment more likely to occur.",
    "Self-Harm": "Content that promotes, encourages, or depicts acts of self-harm, such as suicide, cutting, and eating disorders.",
    "Sexual/Minors": "Sexual content that includes an individual who is under 18 years old.",
    "Hate/Threatening": "Hateful content that also includes violence or serious harm towards the targeted group.",
    "Violence/Graphic": "Violent content that depicts death, violence, or serious physical injury in extreme graphic detail.",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run open-domain policy research for OpenAI moderation domains."
    )
    parser.add_argument("--domain", type=str, default="Sexual/Minors")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model_tag", type=str, default=None)
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--n_research_iters", type=int, default=5)
    parser.add_argument("--n_queries_per_iter", type=int, default=1)
    parser.add_argument("--n_webpages_per_query", type=int, default=10)
    parser.add_argument("--relevance_threshold", type=float, default=0.9)
    parser.add_argument("--chunking_granularity", type=int, default=2000)
    parser.add_argument(
        "--embedding_model_name",
        type=str,
        default="NovaSearch/stella_en_1.5B_v5",
    )
    parser.add_argument("--index_keyphrase_cluster_size", type=int, default=10)
    parser.add_argument("--search_tool", type=str, default="google")
    parser.add_argument("--serper_gl", type=str, default=None)
    parser.add_argument("--serper_hl", type=str, default=None)
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--log_llm_outputs", action="store_true")
    parser.add_argument("--enable_reasoning", action="store_true")
    parser.add_argument("--output_dir", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    print(args)

    model_name = args.model_name
    port = args.port
    llm_client = OpenAI(base_url=f"http://localhost:{port}/v1", api_key="EMPTY")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    exp = args.exp_name
    model_tag = args.model_tag or model_name.replace("/", "_")
    output_dir = args.output_dir or (
        f"./logs/{exp}/{model_tag}/{args.domain.lower().replace('/', '_')}/"
    )
    os.makedirs(output_dir, exist_ok=True)

    research_goal = (
        f'What are the characteristics of sensitive online messages related to "{args.domain}", '
        f'which is defined as "{OPENAI_DOMAIN_TO_BASE_RULE[args.domain]}". Construct a detailed description '
        f"of how these messages look like, and do not care about their detection or effects."
    )
    init_datastore = {
        "rules": [
            {
                "rule": OPENAI_DOMAIN_TO_BASE_RULE[args.domain],
                "sources": "init_rule",
                "rule_relevance_score": 1.0,
            }
        ],
        "examples": [],
    }

    agent_args = {
        "output_dir": output_dir,
        "research_goal": research_goal,
        "in_datastore": init_datastore,
        "n_research_iters": args.n_research_iters,
        "n_queries_per_iter": args.n_queries_per_iter,
        "n_webpages_per_query": args.n_webpages_per_query,
        "relevance_threshold": args.relevance_threshold,
        "model_name": model_name,
        "llm_client": llm_client,
        "tokenizer": tokenizer,
        "chunking_granularity": args.chunking_granularity,
        "embedding_model_name": args.embedding_model_name,
        "index_keyphrase_cluster_size": args.index_keyphrase_cluster_size,
        "search_tool": args.search_tool,
        "serper_gl": args.serper_gl,
        "serper_hl": args.serper_hl,
        "use_cache": not args.no_cache,
        "log_llm_outputs": args.log_llm_outputs,
        "enable_reasoning": args.enable_reasoning,
    }

    args_filtered = {k: v for k, v in agent_args.items() if k not in ["llm_client", "tokenizer"]}
    args_filtered["domain"] = args.domain
    args_filtered["exp_name"] = exp
    args_filtered["model_name"] = model_name
    args_filtered["port"] = port
    args_filtered["model_tag"] = model_tag
    args_filtered["enable_reasoning"] = args.enable_reasoning
    with open(os.path.join(output_dir, "args.json"), "w") as f:
        json.dump(args_filtered, f, indent=4)

    agent = ResearchAgent(**agent_args)
    agent.run_policy_research()


if __name__ == "__main__":
    main()
