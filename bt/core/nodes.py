from typing import Any, Iterable, Mapping


class Node:
    """
    Base node in the portfolio tree.

    A node owns shared portfolio state and child attachment. Nodes are
    constructed unattached and receive a parent only when another node
    attaches them via ``_add_children()``. Concrete strategy and security
    classes implement the execution methods.

    Parameters
    ----------
    name : str
        Node name within its parent scope.
    children : mapping | iterable | None, optional
        Child nodes to attach immediately. Mapping input allows explicit child
        renaming. Child nodes are attached by ownership, not copied.
    """

    def __init__(
        self,
        name: str,
        children: Mapping[str, Any] | Iterable[Any] | None = None,
    ) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("Node name must be a non-empty string.")

        self.name = name
        self.children: dict[str, Node] = {}
        self._childrenv: list[Node] = []
        self._has_strat_children = False
        self._strat_children: list[str] = []
        self.parent: Node | None = None
        self.integer_positions = True

        self.now = 0
        self.inow = 0
        self.last_day = 0

        self._price = 0.0
        self._value = 0.0
        self._weight = 0.0
        self._capital = 0.0

        self._issec = False
        self._add_children(self._normalize_children(children))

    @staticmethod
    def _normalize_children(
        children: Mapping[str, Any] | Iterable[Any] | None,
    ) -> list[Any]:
        """Normalize child input into a concrete list."""
        if children is None:
            return []
        if isinstance(children, Mapping):
            return list(children.items())
        if isinstance(children, (str, bytes)):
            raise TypeError("children must be node objects, not strings.")
        return list(children)

    def __getitem__(self, key: str) -> Node:
        """Return a child node by name."""
        return self.children[key]

    def _add_children(self, children: Iterable[Any] | None) -> None:
        """Attach child nodes to this node with explicit single-parent ownership."""
        if children is None:
            return

        for item in children:
            child_name_override: str | None = None
            child = item

            if isinstance(item, tuple) and len(item) == 2:
                child_name_override, child = item

            if not isinstance(child, Node):
                raise TypeError("children must contain Node instances.")

            if child_name_override is not None:
                if not isinstance(child_name_override, str) or not child_name_override:
                    raise ValueError("Child mapping keys must be non-empty strings.")
                child.name = child_name_override

            if child.name in self.children:
                raise ValueError(f"Child '{child.name}' already exists.")

            if child.parent is not None and child.parent is not self:
                raise ValueError(
                    f"Child '{child.name}' is already attached to '{child.parent.full_name}'."
                )

            child.parent = self
            child.use_integer_positions(self.integer_positions)

            self.children[child.name] = child
            self._childrenv.append(child)

            if getattr(child, "_is_strategy", False):
                self._has_strat_children = True
                self._strat_children.append(child.name)

    def use_integer_positions(self, integer_positions: bool) -> None:
        """Propagate integer-position policy through the subtree."""
        self.integer_positions = bool(integer_positions)
        for child in self._childrenv:
            child.use_integer_positions(self.integer_positions)

    @property
    def prices(self) -> Any:
        """Return the node's price history."""
        raise NotImplementedError()

    @property
    def price(self) -> Any:
        """Return the node's latest price."""
        raise NotImplementedError()

    @property
    def value(self) -> float:
        """Return the current marked value."""
        return self._value

    @property
    def weight(self) -> float:
        """Return the weight relative to the parent."""
        return self._weight

    def setup(self, prices: Any, **kwargs: Any) -> None:
        """Initialize the node against historical input data."""
        raise NotImplementedError()

    def pre_market_update(self, date: Any, inow: int) -> None:
        """Advance the node into the pre-market phase for ``date``."""
        raise NotImplementedError()

    def post_market_update(self) -> None:
        """Finalize node state after the market close."""
        raise NotImplementedError()

    def adjust(
        self,
        amount: float,
        flow: bool = True,
        **kwargs: Any,
    ) -> None:
        """Adjust the node's capital by ``amount``."""
        raise NotImplementedError()

    def allocate(self, amount: float, **kwargs: Any) -> None:
        """Allocate capital to the node."""
        raise NotImplementedError()

    @property
    def members(self) -> list[Node]:
        """Return this node and all descendants in depth-first order."""
        result = [self]
        for child in self._childrenv:
            result.extend(child.members)
        return result

    @property
    def full_name(self) -> str:
        """Return the fully-qualified hierarchical name."""
        if self.parent is None:
            return self.name
        return f"{self.parent.full_name}>{self.name}"

    def __repr__(self) -> str:
        """Return a compact tree-aware representation."""
        return f"<{self.__class__.__name__} {self.full_name}>"

    def to_dot(self, root: bool = True) -> str:
        """Return a DOT representation of this subtree."""

        def display_name(node: Node) -> str:
            return node.name or repr(node)

        edges = "\n".join(
            f'\t"{display_name(self)}" -> "{display_name(child)}"'
            for child in self.children.values()
        )
        below = "\n".join(child.to_dot(False) for child in self.children.values())
        body = "\n".join([edges, below]).rstrip()

        if root:
            return f"digraph {{\n{body}\n}}"
        return body
