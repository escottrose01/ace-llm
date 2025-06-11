from enum import Enum


class Permission(Enum):
    FILESYSTEM = "filesystem_access"
    NETWORK = "network_access"
    DATABASE = "database_access"
    EXECUTION = "execution_access"
