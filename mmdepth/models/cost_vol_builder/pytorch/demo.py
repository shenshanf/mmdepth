from typing import override
from abc import ABC, abstractmethod

class AbstractParent(ABC):
    @abstractmethod
    def speak(self, voice) -> str:
        pass

class Child(AbstractParent):
    @override
    def speak(self, voice, number) -> str:
        print(f"Child speaking: {voice}|{number}")


child = Child()
child.speak(3,5)

