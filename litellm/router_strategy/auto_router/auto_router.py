"""
Auto-Routing Strategy that works with a Semantic Router Config
"""
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from litellm._logging import verbose_router_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.router_strategy.auto_router.template_router import TemplateRouter
from litellm.router_strategy.auto_router.base_template_router import BaseTemplateRouter

import os

if TYPE_CHECKING:
    from semantic_router.routers.base import Route

    from litellm.router import Router
    from litellm.types.router import PreRoutingHookResponse
else:
    Router = Any
    PreRoutingHookResponse = Any
    Route = Any

from datetime import datetime, timedelta

async def save_prompt_response_async(
    prompt: Optional[str],
    cluster_id: Optional[str],
    response: Any,
    duration: Optional[float],
    model: str,
    prisma_client: Optional[Any] = None,
    table_name: str = "auto_router_logs",
) -> bool:
    """
    Save prompt/response/duration/model to the database asynchronously.
    """
    try:
        if prisma_client is None:
            # lazy import to avoid heavy imports at module import time
            from litellm.proxy.utils import get_prisma_client_or_throw  # type: ignore

            prisma_client = get_prisma_client_or_throw(
                "AutoRouter logging"
            )  # may raise if unavailable

        # Normalize duration: accept timedelta, numeric or numeric-string -> store seconds as float
        dur_value = None
        if duration is not None:
            if isinstance(duration, timedelta):
                dur_value = float(duration.total_seconds())
            else:
                try:
                    dur_value = float(duration)
                except Exception:
                    dur_value = None

        # Use a datetime object for Prisma DateTime columns
        record = {
            "prompt": prompt,
            "response": str(response) if response is not None else None,
            "duration": dur_value,
            "model": model,
            "created_at": datetime.utcnow(),
            "cluster_id": cluster_id,
        }

        await prisma_client.db.litellm_autorouterlogs.create(data=record)
        verbose_router_logger.debug(
            f"Saved auto-router log to {table_name}: model={model}, duration={duration}"
        )
        return True
    except Exception as e:
        verbose_router_logger.exception("AutoRouter: failed to save prompt/response to DB")
        print(f"AutoRouter: failed to save prompt/response to DB: {e}")
        return False


class AutoRouter(CustomLogger):
    DEFAULT_AUTO_SYNC_VALUE = "local"
    def __init__(
        self,
        model_name: str,
        default_model: str,
        embedding_model: str,
        litellm_router_instance: "Router",
        auto_router_config_path: Optional[str] = None,
        auto_router_config: Optional[str] = None,
    ):  
        """
        Auto-Router class that uses a semantic router to route requests to the appropriate model.

        Args:
            model_name: The name of the model to use for the auto-router. eg. if model = "auto-router1" then us this router.
            auto_router_config_path: The path to the router config file.
            auto_router_config: The config to use for the auto-router. You can either use this or auto_router_config_path, not both.
            default_model: The default model to use if no route is found.
            embedding_model: The embedding model to use for the auto-router.
            litellm_router_instance: The instance of the LiteLLM Router.
        """
        from semantic_router.routers import SemanticRouter

        self.model_name = model_name
        self.auto_router_config_path: Optional[str] = auto_router_config_path
        self.auto_router_config: Optional[str] = auto_router_config
        self.auto_sync_value = self.DEFAULT_AUTO_SYNC_VALUE
        self.loaded_routes: List[Route] = self._load_semantic_routing_routes()
        self.routelayer: Optional[SemanticRouter] = None
        self.default_model = default_model
        self.embedding_model: str = embedding_model
        self.litellm_router_instance: "Router" = litellm_router_instance

        self.template_routelayer: Optional[BaseTemplateRouter] = TemplateRouter(
            persistence_file=os.path.join("router_templates", f"{model_name}_template_router.bin"),
            metadata_file=os.path.join("router_templates", f"{model_name}_template_router_metadata.pkl"),
        )
    
    def _load_semantic_routing_routes(self) -> List[Route]:
        from semantic_router.routers import SemanticRouter
        if self.auto_router_config_path:
            return SemanticRouter.from_json(self.auto_router_config_path).routes
        elif self.auto_router_config:
            return self._load_auto_router_routes_from_config_json()
        else:
            raise ValueError("No router config provided")
    

    def _load_auto_router_routes_from_config_json(self) -> List[Route]:
        import json

        from semantic_router.routers.base import Route
        
        if self.auto_router_config is None:
            raise ValueError("No auto router config provided")
        auto_router_routes: List[Route] = []
        loaded_config = json.loads(self.auto_router_config)
        for route in loaded_config.get("routes", []):
            auto_router_routes.append(
                Route(
                    name=route.get("name"),
                    description=route.get("description"),
                    utterances=route.get("utterances", []),
                    score_threshold=route.get("score_threshold")
                )
            )
        return auto_router_routes


    async def async_pre_routing_hook(
        self,
        model: str,
        request_kwargs: Dict,
        messages: Optional[List[Dict[str, str]]] = None,
        input: Optional[Union[str, List]] = None,
        specific_deployment: Optional[bool] = False,
    ) -> Optional["PreRoutingHookResponse"]:
        """
        This hook is called before the routing decision is made.

        Used for the litellm auto-router to modify the request before the routing decision is made.
        """
        from semantic_router.routers import SemanticRouter
        from semantic_router.schema import RouteChoice

        from litellm.router_strategy.auto_router.litellm_encoder import (
            LiteLLMRouterEncoder,
        )
        from litellm.types.router import PreRoutingHookResponse
        
        if messages is None:
            # do nothing, return same inputs
            return None
        
        if self.routelayer is None:
            #######################
            # Create the route layer
            #######################
            self.routelayer = SemanticRouter(
                    routes=self.loaded_routes,
                    encoder=LiteLLMRouterEncoder(
                        litellm_router_instance=self.litellm_router_instance,
                        model_name=self.embedding_model,
                    ),
                    auto_sync=self.auto_sync_value,
            )
        user_message: Dict[str, str] = messages[-1]
        message_content: str = user_message.get("content", "")

        # Using Semantic Router to get the route choice
        # route_choice: Optional[Union[RouteChoice, List[RouteChoice]]] = self.routelayer(text=message_content)

        # print(f"AutoRouter: Routing message_content: {message_content}, route_choice: {route_choice}")
        # verbose_router_logger.debug(f"route_choice: {route_choice}")
        # if isinstance(route_choice, RouteChoice):
        #     model = route_choice.name or self.default_model
        # elif isinstance(route_choice, list):
        #     model = route_choice[0].name or self.default_model

        # TODO: handle routes using template based
        model = self.default_model
        if self.template_routelayer is not None:
            template_match = self.template_routelayer.match(message_content)
            if template_match is not None:
                model = template_match.get("target_model", self.default_model)
                verbose_router_logger.info(f"AutoRouter: matched template to cluster {template_match.get('cluster_id')} with target model {model}")
            else:
                verbose_router_logger.info("AutoRouter: no template match found, using default routing")

        return PreRoutingHookResponse(
            model=model,
            messages=messages,
        )
    
    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        """
        Standard CustomLogger callback for successful completions.
        Captures routing decisions, prompts, and responses.
        """
        try:
           
            # Extract prompt from messages
            prompt = None
            if "messages" in kwargs and isinstance(kwargs["messages"], list) and len(kwargs["messages"]) > 0:
                last_message = kwargs["messages"][-1]
                if isinstance(last_message, dict):
                    prompt = last_message.get("content")
                else:
                    prompt = str(last_message)
            
            # Extract response content
            response_content = None
            if response_obj and hasattr(response_obj, 'choices') and len(response_obj.choices) > 0:
                response_content = response_obj.choices[0].message.content if hasattr(response_obj.choices[0], 'message') else str(response_obj.choices[0])
            
            # Log the information
            verbose_router_logger.info("=== AutoRouter Callback - Success Event ===")
            # verbose_router_logger.info(f"Prompt: {prompt}")
            # verbose_router_logger.info(f"Response: {response_content}")
            verbose_router_logger.info(f"Model: {kwargs.get('model', 'unknown')}")
            verbose_router_logger.info(f"Duration: {end_time - start_time if end_time and start_time else 'unknown'}")
            verbose_router_logger.info("==========================================")
            
            # Add to template router history
            if self.template_routelayer and prompt is not None:
                # prompt can be Optional[str] (or other types); ensure we pass a str
                log_message = self.template_routelayer.add_log_message(
                    str(prompt)
                )

                verbose_router_logger.debug("AutoRouter: successfully added prompt to template router history")

                cluster_id = str(log_message.get('cluster_id', ""))

                # Save to database asynchronously
                await save_prompt_response_async(
                    prompt=prompt,
                    cluster_id=cluster_id,
                    response=response_content,
                    duration=(end_time - start_time) if end_time and start_time else None,
                    model=kwargs.get('model', self.default_model),
                )

                verbose_router_logger.debug("AutoRouter: successfully logged success event")
                
           
        except Exception as e:
            verbose_router_logger.exception("AutoRouter: failed to log success event")
