import base64
import os
import time
from dotenv import load_dotenv

from typing import Any, Generator, TypedDict, List, Dict
from langchain.chat_models import init_chat_model

from langgraph.graph import StateGraph, START, END

from src.agent.tools.api_tools import APITools
from src.agent.tools.base_tools import BaseTools
from src.agent.tools.llm_tools import LLMTools
from config import api_config


class State(TypedDict):
    image_path: str
    results: List[Dict]
    top_result: Dict
    samples: List[float]
    sr: int
    description: str
    status: str


class ArtGuide:
    SCORE_THRESHOLD = 0.35

    DURATION_TO_NUM_WORDS = {"short": 100, "medium": 150, "long": 200}

    def __init__(self, config: Dict):
        # LLM initialization
        self.llm = init_chat_model(
            model="openai/gpt-4o",
            model_provider="openai",
            api_key=os.environ["GITHUB_TOKEN"],
            base_url="https://models.github.ai/inference",
        )
        # Configuration
        self.language = config["language"]
        self.speaker = config["speaker"]
        self.n_words = self.DURATION_TO_NUM_WORDS[config["duration"]]

        # Tools
        api_url = api_config['url']
        self.llm_tools = LLMTools(self.llm)
        self.api_tools = APITools(api_url)
        self.utils = BaseTools()

        # Graph initialization
        self.graph_builder = StateGraph(State)
        self.graph = None

    # ========================= NODES =========================

    def search_image_node(self, state: State) -> State:
        # encode image to reach the API
        with open(state["image_path"], 'rb') as image_file:
            image_bytes = image_file.read()
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        results = self.api_tools.search_painting(image_base64)
        return {"results": results}

    def set_top_painting_node(self, state: State) -> State:
        top_painting = self.utils.get_top_result(state["results"])
        if self.language != "en":
            state["top_result"] = self.utils.translate_painting(top_painting, self.language)
        else:
            state["top_result"] = top_painting
        return state

    def deep_search_node(self, state) -> State:
        painting = self.llm_tools.identify_artwork(
            image_path=state["image_path"], language=self.language, n_words=self.n_words
        )

        state["results"].insert(0, painting)  # unnecessary but consistent
        state["top_result"] = painting

        if not painting['title']:
            return {"status": "error"}

        return state

    def generate_description_node(self, state) -> State:
        title = state["top_result"]["title"]

        enriched = self.llm_tools.enrich_painting(
            title=title, language=self.language, n_words=self.n_words
        )
        return {"top_result": enriched | state["top_result"]}

    def generate_speech_node(self, state) -> State:
        description = state["top_result"]["description"]
        audio_data = self.api_tools.synthesize_speech(
            text=description, speaker=self.speaker, language=self.language
        )
        return {
            "samples": audio_data["samples"],
            "sr": audio_data["sr"],
            "top_result": state["top_result"],  # also need in this frontend stage
            "status": "success"
        }

    # ========================= ROUTERS =========================

    def route_deep_research(self, state: State) -> str:
        first_painting = state["results"][0]
        if self.utils.score_meets_threshold(first_painting["score"], self.SCORE_THRESHOLD):
            return "top_painting_selector"
        return "deep_search_image"

    def route_non_painting(self, state: State) -> str:
        if "status" in state:
            return "__end__"
        return "generate_speech"

    # ========================= GRAPH CONFIGURATION =========================

    def create_graph(self):
        self.graph_builder.add_node("search_image", self.search_image_node)
        self.graph_builder.add_node("deep_search_image", self.deep_search_node)
        self.graph_builder.add_node("top_painting_selector", self.set_top_painting_node)
        self.graph_builder.add_node("generate_description", self.generate_description_node)
        self.graph_builder.add_node("generate_speech", self.generate_speech_node)

        self.graph_builder.add_edge(START, "search_image")
        self.graph_builder.add_conditional_edges(
            source="search_image",
            path=self.route_deep_research,
            path_map={
                "deep_search_image": "deep_search_image",
                "top_painting_selector": "top_painting_selector",
            },
        )
        # branch 1:  top_painting_selector >> generate_description (LLM) >> generate_speech >> END
        # if no results are obtained, we go to deep research, no condition for generate_speech
        self.graph_builder.add_edge("top_painting_selector", "generate_description")
        self.graph_builder.add_edge("generate_description", "generate_speech")

        # branch 2: deep_search_image (LLM) >> [generate_speech] >> END
        self.graph_builder.add_conditional_edges(
            source="deep_search_image",
            path=self.route_non_painting,
            path_map={
                "__end__": END,
                "generate_speech": "generate_speech",
            },
        )
        self.graph_builder.add_edge("generate_speech", END)
        self.graph = self.graph_builder.compile()

    def run(self, image_path: str) -> str:
        self.create_graph()

        initial_state: State = {"image_path": image_path}
        state = self.graph.invoke(initial_state)
        return state

    def run_streaming(self, image_path: str) -> Generator[Dict[str, Any], None, None]:
        """
        State updates at each step of the graph execution. Yields dictionaries with node name and
        updated state.
        """
        self.create_graph()

        initial_state: State = {"image_path": image_path}

        for event in self.graph.stream(initial_state):
            for node_name, state_update in event.items():
                yield {"node": node_name, "state": state_update, "timestamp": time.time()}


if __name__ == "__main__":
    load_dotenv()
    agent = ArtGuide({"language": "en", "speaker": "female", "duration": "short"})
    image_path = "/home/afalceto/artguide/img/matrimoni_arnolfini.jpg"
    # image_path = "/home/ubuntu/artguide/img/matrimoni_arnolfini.jpg"
    agent.run(image_path=image_path)
