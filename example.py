"""Пример приложения для управления пользователями."""

import sys

APP_VERSION = "1.0.0"
DB_HOST = "localhost"
DB_NAME = "users_db"


class UserManager:
    """Manages user accounts."""

    def __init__(self):
        self.users = {}

    def add_user(self, username, email):
        if username in self.users:
            print("Error: User '%s' already exists!" % username)
            return False

        if not email or "@" not in email:
            print("Invalid email address provided.")
            return False

        self.users[username] = {"email": email, "role": "user"}
        print("User '%s' has been successfully created." % username)
        return True

    def delete_user(self, username):
        if username not in self.users:
            print("User not found.")
            return False

        del self.users[username]
        print("User has been deleted.")
        return True

    def list_users(self):
        if not self.users:
            print("No users found.")
            return

        print("Registered users:")
        for name, info in self.users.items():
            print("  - {} ({})".format(name, info["email"]))

    def promote_user(self, username):
        if username not in self.users:
            raise ValueError("Cannot promote: user does not exist.")
        self.users[username]["role"] = "admin"
        print("User '{}' is now an administrator.".format(username))


def main():
    manager = UserManager()

    if len(sys.argv) < 2:
        print("Usage: python example.py <command> [args]")
        print("Commands: add, delete, list, promote")
        sys.exit(1)

    command = sys.argv[1]

    if command == "add":
        if len(sys.argv) < 4:
            print("Usage: python example.py add <username> <email>")
            sys.exit(1)
        manager.add_user(sys.argv[2], sys.argv[3])
    elif command == "delete":
        manager.delete_user(sys.argv[2])
    elif command == "list":
        manager.list_users()
    elif command == "promote":
        manager.promote_user(sys.argv[2])
    else:
        print("Unknown command: '%s'" % command)
        print("Available commands: add, delete, list, promote")


if __name__ == "__main__":
    main()
