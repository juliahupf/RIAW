"""
Retrieval-Augmented Generation for the results page.

BaselineRAG is the pipeline we were given, kept unchanged to compare with. ImprovedRAG adds three changes:
1. Richer and cleaner context: product metadata (price, discount, rating, stock, key details) instead of the title
   only, and one entry per product instead of many colour variants of the same item.
2. A refined prompt: a system role with grounding rules, the user's constraints (budget, gender) made explicit,
   a fixed answer format and a low temperature.
3. Handling of "no good products": the answer is decided without the LLM when nothing relevant was retrieved,
   and the products cited by the LLM are checked against the retrieved ones.
"""
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional

from dotenv import load_dotenv

from myapp.search.preprocessing import build_terms

load_dotenv()  # take environment variables from .env

DEFAULT_MODEL = "llama-3.1-8b-instant"
UNAVAILABLE = "RAG is not available. Check your credentials (.env file) or account limits."
PID_PATTERN = re.compile(r"\b[A-Z0-9]{16}\b")


@dataclass
class RAGAnswer:
    text: str
    status: str                      # ok | no_results | no_good_products | ungrounded | unavailable
    used_llm: bool = False
    cited_pids: List[str] = field(default_factory=list)
    context_pids: List[str] = field(default_factory=list)


class _GroqModel:
    """Small wrapper so that both pipelines use the same client and the same model"""

    def __init__(self):
        self.model = os.environ.get("GROQ_MODEL", DEFAULT_MODEL)
        self._client = None

    @property
    def available(self):
        return bool(os.environ.get("GROQ_API_KEY"))

    def complete(self, messages, **kwargs) -> str:
        if self._client is None:
            from groq import Groq
            self._client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        completion = self._client.chat.completions.create(messages=messages, model=self.model, **kwargs)
        return completion.choices[0].message.content


class BaselineRAG:
    """The RAG pipeline provided with the project template (prompt and context unchanged)"""

    PROMPT_TEMPLATE = """
        You are an expert product advisor helping users choose the best option from retrieved e-commerce products.

        ## Instructions:
        1. Identify the single best product that matches the user's request.
        2. Present the recommendation clearly in this format:
        - Best Product: [Product PID] [Product Name]
        - Why: [Explain in plain language why this product is the best fit, referring to specific attributes like price, features, quality, or fit to user’s needs.]
        3. If there is another product that could also work, mention it briefly as an alternative.
        4. If no product is a good fit, return ONLY this exact phrase:
        "There are no good products that fit the request based on the retrieved results."

        ## Retrieved Products:
        {retrieved_results}

        ## User Request:
        {user_query}

        ## Output Format:
        - Best Product: ...
        - Why: ...
        - Alternative (optional): ...
    """

    def __init__(self, model: Optional[_GroqModel] = None):
        self.llm = model or _GroqModel()

    def generate_response(self, user_query: str, retrieved_results: list, top_N: int = 20, **_) -> RAGAnswer:
        documents = [getattr(r, "document", r) for r in retrieved_results[:top_N]]
        if not self.llm.available:
            return RAGAnswer(UNAVAILABLE, "unavailable")
        formatted_results = "\n".join(f"- PID: {doc.pid}, Title: {doc.title}" for doc in documents)
        prompt = self.PROMPT_TEMPLATE.format(retrieved_results=formatted_results, user_query=user_query)
        try:
            text = self.llm.complete([{"role": "user", "content": prompt}])
        except Exception as e:
            print(f"Error during RAG generation: {e}")
            return RAGAnswer(UNAVAILABLE, "unavailable")
        return RAGAnswer(text, "ok", used_llm=True, cited_pids=PID_PATTERN.findall(text),
                         context_pids=[doc.pid for doc in documents])


class ImprovedRAG:
    """RAG with metadata context, a grounded prompt and handling of queries without good products"""

    SYSTEM_PROMPT = """You are a shopping assistant for a fashion e-commerce website.
You recommend products ONLY from the list of retrieved products given by the user message.
Rules:
- Never invent products, prices or features. Use only the information in the list.
- Always refer to a product by its title followed by its PID in brackets, for example: Running Shoes For Men (Blue) [SHOEWQKVSGFGXPQF].
- Respect every constraint of the request (gender, colour, type of product, budget). A product that breaks a constraint is not a good fit, even if it ranks high.
- Prefer products that are in stock, then better rated and better priced.
- If none of the products fits the request, say so in one sentence and suggest how to rephrase the search.
- Answer in English, in at most 120 words, with this format:
**Best choice:** <title> [<PID>] - <one sentence with the reasons, mentioning price and rating>
**Also consider:** <title> [<PID>] - <short reason>   (one or two lines, optional)
**Tip:** <one short sentence to help the user decide or refine the search>"""

    USER_TEMPLATE = """User request: "{query}"
{constraints}
Retrieved products ({count}, ordered by our search engine):
{products}"""

    def __init__(self, model: Optional[_GroqModel] = None, context_size: int = 8, min_title_match: float = 0.5):
        """
        :param context_size: products sent to the LLM. Fewer, cleaner products give a shorter prompt and less confusion
        :param min_title_match: share of the query terms that the best product title must contain to call the LLM
        """
        self.llm = model or _GroqModel()
        self.context_size = context_size
        self.min_title_match = min_title_match

    def _select_context(self, documents: list) -> list:
        """Top products without colour variants of the same item (same title and brand)"""
        seen, selected = set(), []
        for doc in documents:
            key = (" ".join(doc.title.lower().split()), (doc.brand or "").lower())
            if key not in seen:
                seen.add(key)
                selected.append(doc)
            if len(selected) == self.context_size:
                break
        return selected

    @staticmethod
    def _describe(doc) -> str:
        details = doc.product_details or {}
        useful = ("Ideal For", "Color", "Fabric", "Type", "Occasion", "Pattern", "Fit", "Sleeve", "Hooded",
                  "Outer Material", "Neck")
        attributes = "; ".join(f"{k}: {details[k]}" for k in useful if details.get(k))
        price = f"Rs {doc.selling_price:.0f}" if doc.selling_price is not None else "unknown price"
        if doc.discount:
            price += f" ({doc.discount:.0f}% off)"
        rating = f"{doc.average_rating}/5" if doc.average_rating is not None else "no rating"
        stock = "OUT OF STOCK" if doc.out_of_stock else "in stock"
        description = " ".join((doc.description or "").split())[:160]
        line = f"- [{doc.pid}] {' '.join(doc.title.split())} | brand: {doc.brand or 'unknown'} | {price} | {rating} | {stock}"
        if attributes:
            line += f" | {attributes}"
        if description:
            line += f" | {description}"
        return line

    def _title_match(self, query_terms: List[str], doc) -> float:
        unique = set(query_terms)
        return len(unique & set(build_terms(doc.title))) / len(unique) if unique else 0.0

    def generate_response(self, user_query: str, retrieved_results: list, query_terms: Optional[List[str]] = None,
                          max_price: Optional[float] = None, relaxed: bool = False, **_) -> RAGAnswer:
        """
        :param retrieved_results: ranked SearchHit or Document objects
        :param query_terms: processed query terms (to check how well the products match)
        :param max_price: budget found in the query or in the filters
        :param relaxed: True when no product contained all the query terms
        """
        documents = [getattr(r, "document", r) for r in retrieved_results]
        query_terms = query_terms if query_terms is not None else build_terms(user_query)

        # 3a. No good products: decided without spending an LLM call (and without the risk of an invented answer)
        if not documents:
            return RAGAnswer(f'We could not find any product for "{user_query}". Try fewer or more general words, '
                             "for example the type of product and the gender.", "no_results")
        context = self._select_context(documents)
        best_match = max(self._title_match(query_terms, doc) for doc in context)
        if relaxed or best_match < self.min_title_match:
            closest = context[0]
            return RAGAnswer(f'No product matches all of "{user_query}". The closest result is '
                             f"{' '.join(closest.title.split())} [{closest.pid}], but it does not cover the whole "
                             "request. Try removing a word or using a more general term.",
                             "no_good_products", context_pids=[d.pid for d in context])

        if not self.llm.available:
            return RAGAnswer(UNAVAILABLE, "unavailable", context_pids=[d.pid for d in context])

        # 1 and 2. Rich context and grounded prompt
        constraints = f"Budget: at most Rs {max_price:.0f}." if max_price is not None else ""
        message = self.USER_TEMPLATE.format(query=user_query, constraints=constraints, count=len(context),
                                            products="\n".join(self._describe(doc) for doc in context))
        try:
            text = self.llm.complete([{"role": "system", "content": self.SYSTEM_PROMPT},
                                      {"role": "user", "content": message}],
                                     temperature=0.2, max_tokens=300)
        except Exception as e:
            print(f"Error during RAG generation: {e}")
            return RAGAnswer(UNAVAILABLE, "unavailable", context_pids=[d.pid for d in context])

        # 3b. Grounding check: every product cited must be one of the products we gave to the model
        context_pids = [doc.pid for doc in context]
        cited = PID_PATTERN.findall(text)
        if any(pid not in context_pids for pid in cited):
            best = context[0]
            return RAGAnswer(f"**Best choice:** {' '.join(best.title.split())} [{best.pid}] - top result of the search "
                             "(the generated summary cited a product that was not retrieved, so it was discarded).",
                             "ungrounded", used_llm=True, cited_pids=cited, context_pids=context_pids)
        return RAGAnswer(text, "ok", used_llm=True, cited_pids=cited, context_pids=context_pids)
