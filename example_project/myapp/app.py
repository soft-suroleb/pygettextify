"""Simple task manager application."""

import sys
from datetime import datetime


APP_NAME = "TaskMaster"
VERSION = "1.0.0"
DB_PATH = "data/tasks.db"


class Task:
    def __init__(self, title, description="", priority="medium"):
        self.title = title
        self.description = description
        self.priority = priority
        self.created_at = datetime.now()
        self.done = False

    def __str__(self):
        status = "Done" if self.done else "Pending"
        return f"{self.title} [{status}]"

    def mark_done(self):
        self.done = True
        print("Task marked as completed successfully!")

    def mark_undone(self):
        self.done = False
        print("Task reopened.")


class TaskManager:
    def __init__(self):
        self.tasks = []

    def add_task(self, title, description="", priority="medium"):
        if not title.strip():
            print("Error: Task title cannot be empty.")
            return None

        valid_priorities = ["low", "medium", "high", "critical"]
        if priority not in valid_priorities:
            print("Invalid priority. Please choose from: low, medium, high, critical.")
            return None

        task = Task(title, description, priority)
        self.tasks.append(task)
        print("Task added successfully!")
        return task

    def remove_task(self, index):
        if index < 0 or index >= len(self.tasks):
            print("Error: Invalid task number.")
            return False
        removed = self.tasks.pop(index)
        print(f"Task '{removed.title}' has been removed.")
        return True

    def list_tasks(self):
        if not self.tasks:
            print("No tasks found. Your task list is empty.")
            return

        print("Your tasks:")
        print("-" * 40)
        for i, task in enumerate(self.tasks, 1):
            marker = "x" if task.done else " "
            print(f"  [{marker}] {i}. {task}")
        print("-" * 40)
        total = len(self.tasks)
        done = sum(1 for t in self.tasks if t.done)
        print(f"Total: {total} tasks, {done} completed, {total - done} remaining.")

    def search_tasks(self, query):
        results = [t for t in self.tasks if query.lower() in t.title.lower()]
        if not results:
            print("No tasks matching your search were found.")
            return []
        print(f"Found {len(results)} matching tasks:")
        for task in results:
            print(f"  - {task}")
        return results
