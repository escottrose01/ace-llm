from pydantic import BaseModel


class TypewriterOutputType(BaseModel):
    result: str
    success: bool


class ListUserIdsOutputType(BaseModel):
    result: list[int]


class GetUserNameOutputType(BaseModel):
    result: str


class UserItem(BaseModel):
    id: int
    name: str


class FindUsersByNameOutputType(BaseModel):
    result: list[UserItem]


class LocationItem(BaseModel):
    id: int
    city: str


class FindLocationsByNameOutputType(BaseModel):
    result: list[LocationItem]


class FoodItem(BaseModel):
    id: int
    name: str


class FindFoodsByNameOutputType(BaseModel):
    result: list[FoodItem]


class GetUserEmailOutputType(BaseModel):
    result: str


class GetUserLocationOutputType(BaseModel):
    result: int


class GetUserFavoriteColorOutputType(BaseModel):
    result: str


class GetUserFavoriteFoodsOutputType(BaseModel):
    result: list[int]


class GetWeatherAtLocationOutputType(BaseModel):
    result: str


class GetCityForLocationOutputType(BaseModel):
    result: str


class GetCurrentTimeForLocationOutputType(BaseModel):
    result: str


class GetCurrentWeatherForLocationOutputType(BaseModel):
    result: str


class GetFoodNameOutputType(BaseModel):
    result: str


class GetFoodCaloriesOutputType(BaseModel):
    result: str


class GetFoodAllergicIngredientsOutputType(BaseModel):
    result: list[str]


class GetCurrentUserIdOutputType(BaseModel):
    result: int


TYPE_PATCH = {
    # Typewriter tool outputs
    "type_letter_output": TypewriterOutputType,
    **{f"{letter}_output": TypewriterOutputType for letter in "abcdefghijklmnopqrstuvwxyz"},
    # Relational data tool outputs
    "list_user_ids_output": ListUserIdsOutputType,
    "get_user_name_output": GetUserNameOutputType,
    "find_users_by_name_output": FindUsersByNameOutputType,
    "find_locations_by_name_output": FindLocationsByNameOutputType,
    "find_foods_by_name_output": FindFoodsByNameOutputType,
    "get_user_email_output": GetUserEmailOutputType,
    "get_user_location_output": GetUserLocationOutputType,
    "get_user_favorite_color_output": GetUserFavoriteColorOutputType,
    "get_user_favorite_foods_output": GetUserFavoriteFoodsOutputType,
    "get_weather_at_location_output": GetWeatherAtLocationOutputType,
    "get_city_for_location_output": GetCityForLocationOutputType,
    "get_current_time_for_location_output": GetCurrentTimeForLocationOutputType,
    "get_current_weather_for_location_output": GetCurrentWeatherForLocationOutputType,
    "get_food_name_output": GetFoodNameOutputType,
    "get_food_calories_output": GetFoodCaloriesOutputType,
    "get_food_allergic_ingredients_output": GetFoodAllergicIngredientsOutputType,
    "get_current_user_id_output": GetCurrentUserIdOutputType,
}
