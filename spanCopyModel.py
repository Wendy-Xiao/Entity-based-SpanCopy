from shutil import Error
from torch import nn
import torch
from transformers import PegasusForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence
import pdb


def shift_tokens_right(
    input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int
):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


class SpanPointerNetwork(PegasusForConditionalGeneration):
    def __init__(self, config, entity_limit=100, global_relevance=False):
        super().__init__(config)
        self.config = config
        self.config.mixed_vocab_size = config.vocab_size + entity_limit
        hidden_size = config.d_model
        self.p_copy_layer = nn.Sequential(nn.Linear(hidden_size, 1), nn.Sigmoid())
        # self.p_gen_layer = nn.Sequential(nn.Linear(hidden_size, 1), nn.Sigmoid())
        self.copy_attenion_q = nn.Linear(hidden_size, hidden_size)
        self.copy_attenion_k = nn.Linear(hidden_size, hidden_size)
        self.global_relevance = global_relevance
        if global_relevance:
            self.global_relevance_layer = nn.Sequential(
                nn.Linear(hidden_size, 1), nn.Sigmoid()
            )

    def embed_with_entities(self, output_ids, entity_mapping):
        # output_ids: batch*len_output_ids
        # entity mapping: (batch, len_output_ids, updated_len_output_ids)
        # for any one example, e: n*m, sum(e[:,j])=1
        # token_decoder_embeds: batch*len_output_ids*d_model
        token_decoder_embeds = self.model.shared(output_ids)
        updated_decoder_embeds = torch.bmm(
            token_decoder_embeds.transpose(1, 2), entity_mapping
        ).transpose(1, 2)
        return updated_decoder_embeds

    def forward(
        self,
        input_ids,
        input_entity_mapping,
        output_entity_mapping=None,
        decoder_inputs_embeds=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        entity_keys=None,
        entity_mask=None,
        return_pgen=False,
        gr=None,
        return_gr=False,
    ):
        if decoder_inputs_embeds is None:
            if decoder_input_ids is not None:
                decoder_inputs_embeds = self.embed_with_entities(
                    decoder_input_ids, output_entity_mapping
                )
            else:
                raise Error("No decoding input")

        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )
        modelOutput = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=None,  # use decoder_input_embeds instead
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
        )
        lm_logits = self.lm_head(modelOutput.last_hidden_state) + self.final_logits_bias
        # (batch*len_output*voc)
        decoder_logits = lm_logits
        # (batch*len_output*d_model)
        decoder_hidden_states_last = modelOutput.last_hidden_state
        if entity_keys is None:
            # (batch*len_input*d_model)
            encoder_hidden_states_last = modelOutput.encoder_last_hidden_state
            a = encoder_hidden_states_last.permute(0, 2, 1)
            b = input_entity_mapping
            # (batch*num_entities*d_model)
            entities = torch.bmm(a, b).permute(0, 2, 1)
            entity_mask = torch.zeros(
                entities.shape[0], entities.shape[1], device=self.device
            )
            entity_mask[b.sum(1).nonzero(as_tuple=True)] = 1
            entity_keys = self.copy_attenion_k(entities)
            if self.global_relevance and gr is None:
                # (batch*num_entities)
                gr = self.global_relevance_layer(entities).squeeze(-1)
                # gr *= entity_mask
        # (batch*len_output*num_entities)
        logits_copy = torch.bmm(
            entity_keys,
            self.copy_attenion_q(decoder_hidden_states_last).transpose(1, 2),
        ).transpose(1, 2)
        if self.global_relevance:
            expanded_gr = gr[:, None, :].expand_as(logits_copy)
            logits_copy *= expanded_gr
        expanded_entity_mask = entity_mask[:, None, :].expand_as(logits_copy)
        logits_copy *= expanded_entity_mask
        # expanded_entity_mask=(1-expanded_entity_mask)*-1e5
        # logits_copy += expanded_entity_mask
        # (batch*len_output*1)
        p_copy = self.p_copy_layer(decoder_hidden_states_last)
        lm_logits = torch.cat([(1 - p_copy) * decoder_logits, p_copy * logits_copy], -1)
        # p_gen = self.p_gen_layer(decoder_hidden_states_last)
        # lm_logits = torch.cat([p_gen * decoder_logits, (1 - p_gen) * logits_copy], -1)
        # pdb.set_trace()
        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(
                lm_logits.view(-1, self.config.vocab_size), labels.view(-1)
            )
        # if torch.isnan(lm_logits).any():
        #     pdb.set_trace()
        output = Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=modelOutput.past_key_values,
            decoder_hidden_states=modelOutput.decoder_hidden_states,
            decoder_attentions=modelOutput.decoder_attentions,
            cross_attentions=modelOutput.cross_attentions,
            encoder_last_hidden_state=modelOutput.encoder_last_hidden_state,
            encoder_hidden_states=modelOutput.encoder_hidden_states,
            encoder_attentions=modelOutput.encoder_attentions,
        )
        # pdb.set_trace()
        if (not return_pgen) and (not return_gr):
            return output
        out = {}
        out["output"] = output
        if return_pgen:
            out["p_copy"] = p_copy
        if return_gr:
            out["gr"] = gr
        return out

    def build_entity_mapping(self, decoder_input_ids, input_ids, input_entity_mapping):
        entity_mapping = []
        decoder_single_ids = []
        for i_batch in range(decoder_input_ids.shape[0]):
            single_token_ids = []
            token_entity_map = [[], []]
            for i_token in range(decoder_input_ids[i_batch].shape[0]):
                if decoder_input_ids[i_batch][i_token] > self.config.vocab_size:
                    entity_id = (
                        decoder_input_ids[i_batch][i_token] - self.config.vocab_size
                    )
                    positions = input_entity_mapping[i_batch, :, entity_id].nonzero()
                    token_id = input_ids[positions]
                    token_entity_map[0].extend(
                        [len(single_token_ids) + i for i in range(len(token_id))]
                    )
                    token_entity_map[1].extend([i_token for _ in range(len(token_id))])
                    single_token_ids.extend(token_id)
                else:
                    token_entity_map[0].append(len(single_token_ids))
                    token_entity_map[1].append(i_token)
                    single_token_ids.append(decoder_input_ids[i_batch][i_token])
            mapping = torch.zeros(
                (len(single_token_ids), len(decoder_input_ids[i_batch]))
            )
            mapping[token_entity_map] = 1
            mapping = mapping / mapping.sum(dim=0)
            entity_mapping.append(mapping)
            decoder_single_ids.append(torch.tensor(single_token_ids, dtype=torch.long))
        output_mapping = torch.zeros(
            (
                decoder_input_ids.shape[0],
                max([e.shape[0] for e in entity_mapping]),
                max([e.shape[1] for e in entity_mapping]),
            )
        )
        for i_batch in range(len(entity_mapping)):
            output_mapping[
                i_batch,
                : entity_mapping[i_batch].shape[0],
                : entity_mapping[i_batch].shape[1],
            ] = entity_mapping[i_batch]
        decoder_single_ids = pad_sequence(
            decoder_single_ids, batch_first=True, padding_value=self.config.pad_token_id
        )
        return decoder_single_ids, output_mapping

    def build_decoder_embedding_single(
        self, decoder_input_ids, input_ids, input_entity_mapping
    ):
        decoder_embed_single = []
        # decoder_input_ids: batch_size * 1
        for i_batch in range(decoder_input_ids.shape[0]):
            token_id = decoder_input_ids[i_batch][0]
            if token_id >= self.config.vocab_size:
                entity_id = decoder_input_ids[i_batch][0] - self.config.vocab_size
                positions = input_entity_mapping[i_batch, :, entity_id].nonzero(
                    as_tuple=True
                )[0]
                token_id = input_ids[i_batch, positions]
            else:
                token_id = torch.tensor(
                    [token_id], dtype=torch.long, device=token_id.device
                )
            embeddings = self.model.shared(token_id)
            decoder_embed_single.append(embeddings.mean(dim=0))
        decoder_embed_single = torch.stack(decoder_embed_single, dim=0).unsqueeze(1)
        return decoder_embed_single

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        decoder_helper_encoder_input_ids=None,
        decoder_input_entity_mapping=None,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        entity_keys=None,
        entity_mask=None,
        decoder_gr=None,
        **kwargs,
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]
            decoder_embed_single = self.build_decoder_embedding_single(
                decoder_input_ids,
                decoder_helper_encoder_input_ids,
                decoder_input_entity_mapping,
            )
            decoder_input_ids = None
            decoder_entity_mapping = None
        elif decoder_input_ids.shape[1] == 1:
            decoder_embed_single = self.build_decoder_embedding_single(
                decoder_input_ids,
                decoder_helper_encoder_input_ids,
                decoder_input_entity_mapping,
            )
            decoder_input_ids = None
            decoder_entity_mapping = None
        else:
            decoder_input_ids, decoder_entity_mapping = self.build_entity_mapping(
                decoder_input_ids,
                decoder_helper_encoder_input_ids,
                decoder_input_entity_mapping,
            )
            decoder_embed_single = None
            decoder_input_ids.to(self.device)
            decoder_entity_mapping.to(self.device)
        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "input_entity_mapping": decoder_input_entity_mapping,
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
            "decoder_input_ids": decoder_input_ids,
            "output_entity_mapping": decoder_entity_mapping,
            "decoder_inputs_embeds": decoder_embed_single,
            "entity_keys": entity_keys,
            "entity_mask": entity_mask,
            "gr": decoder_gr,
        }

    def _prepare_encoder_decoder_kwargs_for_generation(self, input_ids, model_kwargs):
        if "encoder_outputs" not in model_kwargs:
            # retrieve encoder hidden states
            encoder = self.get_encoder()
            encoder_kwargs = {
                argument: value
                for argument, value in model_kwargs.items()
                if not (
                    argument.startswith("decoder_") or argument.startswith("cross_attn")
                )
            }
            ModelOutput = encoder(input_ids, return_dict=True, **encoder_kwargs)
            model_kwargs["encoder_outputs"] = ModelOutput
            encoder_hidden_states_last = ModelOutput.last_hidden_state
            a = encoder_hidden_states_last.permute(0, 2, 1)
            b = model_kwargs["decoder_input_entity_mapping"]
            entities = torch.bmm(a, b).permute(0, 2, 1)
            entity_mask = torch.zeros(
                entities.shape[0], entities.shape[1], device=self.device
            )
            entity_mask[b.sum(1).nonzero(as_tuple=True)] = 1
            entity_keys = self.copy_attenion_k(entities)
            model_kwargs["entity_keys"] = entity_keys
            model_kwargs["entity_mask"] = entity_mask
            if self.global_relevance and model_kwargs['decoder_gr'] is None:
                # (batch*num_entities)
                gr = self.global_relevance_layer(entities).squeeze(-1)
                model_kwargs["decoder_gr"] = gr

        return model_kwargs

    @staticmethod
    def _expand_inputs_for_generation(
        input_ids,
        expand_size=1,
        is_encoder_decoder=False,
        attention_mask=None,
        encoder_outputs=None,
        **model_kwargs,
    ):
        expanded_return_idx = (
            torch.arange(input_ids.shape[0])
            .view(-1, 1)
            .repeat(1, expand_size)
            .view(-1)
            .to(input_ids.device)
        )
        input_ids = input_ids.index_select(0, expanded_return_idx)

        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = token_type_ids.index_select(
                0, expanded_return_idx
            )

        if attention_mask is not None:
            model_kwargs["attention_mask"] = attention_mask.index_select(
                0, expanded_return_idx
            )

        if is_encoder_decoder:
            if encoder_outputs is None:
                raise ValueError(
                    "If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined."
                )
            encoder_outputs[
                "last_hidden_state"
            ] = encoder_outputs.last_hidden_state.index_select(
                0, expanded_return_idx.to(encoder_outputs.last_hidden_state.device)
            )
            model_kwargs["encoder_outputs"] = encoder_outputs
            model_kwargs["entity_keys"] = model_kwargs["entity_keys"].index_select(
                0, expanded_return_idx
            )
            model_kwargs["entity_mask"] = model_kwargs["entity_mask"].index_select(
                0, expanded_return_idx
            )
            if "gr" in model_kwargs:
                model_kwargs["gr"] = model_kwargs["gr"].index_select(
                    0, expanded_return_idx
                )
        return input_ids, model_kwargs

