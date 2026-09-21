"""Dataset specs and loading.

Each spec fixes the label id -> description map. The same descriptions are
used for JEV ``criteria``, the LLM prompt, and zero-shot NLI hypotheses, so
every zero-shot model sees identical label semantics.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Iterable


@dataclass
class Example:
    text: str
    label: str


@dataclass
class DatasetSpec:
    hf_name: str
    eval_split: str
    text_field: str
    labels: dict[str, str]  # label id -> description, in HF label-index order
    train_split: str = "train"


@dataclass
class Dataset:
    name: str
    labels: dict[str, str]
    examples: list[Example]
    train: list[Example] = field(default_factory=list)

    @property
    def label_ids(self) -> list[str]:
        return list(self.labels)


# Official Banking77 intents in HF label-index order.
BANKING77_RAW = [
    "activate_my_card", "age_limit", "apple_pay_or_google_pay", "atm_support",
    "automatic_top_up", "balance_not_updated_after_bank_transfer",
    "balance_not_updated_after_cheque_or_cash_deposit", "beneficiary_not_allowed",
    "cancel_transfer", "card_about_to_expire", "card_acceptance", "card_arrival",
    "card_delivery_estimate", "card_linking", "card_not_working",
    "card_payment_fee_charged", "card_payment_not_recognised",
    "card_payment_wrong_exchange_rate", "card_swallowed", "cash_withdrawal_charge",
    "cash_withdrawal_not_recognised", "change_pin", "compromised_card",
    "contactless_not_working", "country_support", "declined_card_payment",
    "declined_cash_withdrawal", "declined_transfer",
    "direct_debit_payment_not_recognised", "disposable_card_limits",
    "edit_personal_details", "exchange_charge", "exchange_rate", "exchange_via_app",
    "extra_charge_on_statement", "failed_transfer", "fiat_currency_support",
    "get_disposable_virtual_card", "get_physical_card", "getting_spare_card",
    "getting_virtual_card", "lost_or_stolen_card", "lost_or_stolen_phone",
    "order_physical_card", "passcode_forgotten", "pending_card_payment",
    "pending_cash_withdrawal", "pending_top_up", "pending_transfer", "pin_blocked",
    "receiving_money", "Refund_not_showing_up", "request_refund",
    "reverted_card_payment?", "supported_cards_and_currencies", "terminate_account",
    "top_up_by_bank_transfer_charge", "top_up_by_card_charge",
    "top_up_by_cash_or_cheque", "top_up_failed", "top_up_limits", "top_up_reverted",
    "topping_up_by_card", "transaction_charged_twice", "transfer_fee_charged",
    "transfer_into_account", "transfer_not_received_by_recipient", "transfer_timing",
    "unable_to_verify_identity", "verify_my_identity", "verify_source_of_funds",
    "verify_top_up", "virtual_card_not_working", "visa_or_mastercard",
    "why_verify_identity", "wrong_amount_of_cash_received",
    "wrong_exchange_rate_for_cash_withdrawal",
]


def _bank_id(raw: str) -> str:
    return raw.lower().replace("?", "")


def _bank_desc(raw: str) -> str:
    return "Customer asks about: " + raw.lower().replace("?", "").replace("_", " ")


DATASETS: dict[str, DatasetSpec] = {
    "sst2": DatasetSpec(
        "stanfordnlp/sst2", "validation", "sentence",
        {
            "negative": "The sentence expresses negative sentiment about the movie.",
            "positive": "The sentence expresses positive sentiment about the movie.",
        },
    ),
    "agnews": DatasetSpec(
        "fancyzhx/ag_news", "test", "text",
        {
            "world": "World news: politics, governments, international affairs, conflicts.",
            "sports": "Sports news: games, athletes, teams, matches, results.",
            "business": "Business news: companies, markets, economy, finance, deals.",
            "sci_tech": "Science and technology news: research, software, hardware, internet.",
        },
    ),
    "banking77": DatasetSpec(
        "legacy-datasets/banking77", "test", "text",
        {_bank_id(r): _bank_desc(r) for r in BANKING77_RAW},
    ),
}

Loader = Callable[[str, str], Iterable[dict]]


def _hf_loader(hf_name: str, split: str):
    from datasets import load_dataset

    return load_dataset(hf_name, split=split)


def _to_examples(rows: Iterable[dict], spec: DatasetSpec) -> list[Example]:
    ids = list(spec.labels)
    return [Example(text=r[spec.text_field], label=ids[int(r["label"])]) for r in rows]


def load(
    name: str, n: int, seed: int, train_cap: int, loader: Loader | None = None
) -> Dataset:
    spec = DATASETS[name]
    loader = loader or _hf_loader
    rng = random.Random(seed)
    ev = _to_examples(loader(spec.hf_name, spec.eval_split), spec)
    rng.shuffle(ev)
    tr = _to_examples(loader(spec.hf_name, spec.train_split), spec)
    rng.shuffle(tr)
    return Dataset(name, spec.labels, ev[:n], tr[:train_cap])
