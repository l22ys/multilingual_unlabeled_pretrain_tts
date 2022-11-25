'''
Copyright 2018 The Hugging Face team

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

from transformers import Wav2Vec2ForPreTraining, Wav2Vec2Model
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2EncoderStableLayerNorm
from transformers.deepspeed import is_deepspeed_zero3_enabled
import numpy as np
import torch

class ReducedWav2Vec2EncoderStableLayerNorm(Wav2Vec2EncoderStableLayerNorm):
    def __init__(self, *args, **kwargs):
        super(ReducedWav2Vec2EncoderStableLayerNorm, self).__init__(*args, **kwargs)
        
    def leave_only_1layer(self):
        for i in range(1,24):
            del self.layers._modules[str(i)]
        # print(len(self.layers))

    def leave_only_12layers(self):
        for i in range(12,24):
            del self.layers._modules[str(i)]
        # print(len(self.layers))

    def leave_only_15layers(self):
        for i in range(15,24):
            del self.layers._modules[str(i)]

    def freeze_except_last_layers(self, num): # num is 'starting index of encoder block' for not freezing
        for param in self.parameters():
            param.requires_grad = False
        
        for i in range(num, 24):
            for param in self.layers._modules[str(i)].parameters():
                param.requires_grad = True
        
    # copied and modified from https://github.com/huggingface/transformers/blob/main/src/transformers/models/wav2vec2/modeling_wav2vec2.py
    def reduced_forward(self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True):

        all_hidden_states = () if output_hidden_states else None 
        all_self_attentions = () if output_attentions else None # None

        if attention_mask is not None:
            # make sure padded tokens are not attended to
            hidden_states[~attention_mask] = 0

            # extend attention_mask
            attention_mask = (1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask.expand(
                attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1]
            )

        position_embeddings = self.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.dropout(hidden_states)

        deepspeed_zero3_is_enabled = is_deepspeed_zero3_enabled() # False

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        for layer in self.layers:
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = np.random.uniform(0, 1)

            skip_the_layer = True if self.training and (dropout_probability < self.config.layerdrop) else False
            
            
            if not skip_the_layer or deepspeed_zero3_is_enabled:
                # under deepspeed zero3 all gpus must run in sync
                # XXX: could optimize this like synced_gpus in generate_utils but not sure if it's worth the code complication
                if self.gradient_checkpointing and self.training:
                    # create gradient checkpointing function
                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(layer),
                        hidden_states,
                        attention_mask,
                    )
                else:
                    layer_outputs = layer(
                        hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
                    )
                hidden_states = layer_outputs[0] 

            if skip_the_layer: # False
                layer_outputs = (None, None)

            if output_attentions: # False
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

            if output_hidden_states: 
                all_hidden_states = all_hidden_states + (hidden_states,)

        return all_hidden_states


class ReducedWav2Vec2Model(Wav2Vec2Model):
    def __init__(self, *args, **kwargs):
        super(ReducedWav2Vec2Model, self).__init__(*args, **kwargs)
        self.encoder = ReducedWav2Vec2EncoderStableLayerNorm(*args, **kwargs)

    def leave_only_1layer(self):
        self.encoder.leave_only_1layer()

    def leave_only_12layers(self):
        self.encoder.leave_only_12layers() 

    def leave_only_15layers(self):
        self.encoder.leave_only_15layers()

    def freeze_except_last_layers(self, num):
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        for param in self.feature_projection.parameters():
            param.requires_grad = False
        self.encoder.freeze_except_last_layers(num)
    
    # copied and modified from https://github.com/huggingface/transformers/blob/main/src/transformers/models/wav2vec2/modeling_wav2vec2.py
    def reduced_forward(self,
        input_values,
        attention_mask=None,
        mask_time_indices=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # False

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # True

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # True

        extract_features = self.feature_extractor(input_values) 
        extract_features = extract_features.transpose(1, 2) ## torch.Tensor
        # [B, seq_leng, dim]

        if attention_mask is not None:
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask, add_adapter=False
            )

        hidden_states, _ = self.feature_projection(extract_features)
        # hidden_states : [batch_size, sequence_length, hidden_size]

        hidden_states = self._mask_hidden_states(
            hidden_states, mask_time_indices=mask_time_indices, attention_mask=attention_mask
        )

        all_hidden_states = self.encoder.reduced_forward(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,)

        return all_hidden_states, attention_mask

class ReducedWav2Vec2ForPreTraining(Wav2Vec2ForPreTraining):
    def __init__(self, *args, **kwargs):
        super(ReducedWav2Vec2ForPreTraining, self).__init__(*args, **kwargs)
        self.wav2vec2 = ReducedWav2Vec2Model(*args, **kwargs)
    
    def train(self, mode: bool=True):
        super(ReducedWav2Vec2ForPreTraining, self).train(mode)
        self.wav2vec2.eval()

    def eval(self):
        super(ReducedWav2Vec2ForPreTraining, self).eval()
    
    def leave_only_1layer(self):
        self.wav2vec2.leave_only_1layer()
        del self.quantizer
        del self.project_hid
        del self.project_q

    def leave_only_12layers(self):
        self.wav2vec2.leave_only_12layers()
        del self.quantizer
        del self.project_hid
        del self.project_q

    def leave_only_15layers(self):
        self.wav2vec2.leave_only_15layers()
        del self.quantizer
        del self.project_hid
        del self.project_q

    def leave_only_24layers(self):
        del self.quantizer
        del self.project_hid
        del self.project_q

    def freeze_except_last_layers(self, num): # num is 'starting index of encoder block' for not freezing
        self.wav2vec2.freeze_except_last_layers(num)

    # copied and modified from https://github.com/huggingface/transformers/blob/main/src/transformers/models/wav2vec2/modeling_wav2vec2.py
    def reduced_forward(self,
        input_values,
        attention_mask=None,
        mask_time_indices=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,):
        '''
        input_values - [B, t] : batch of raw waveforms
        input_values_lengths - [B] : batch of lengths of waveforms
        attention_mask - [B, t] : mask corresponding to input_values_lengths
        output_attentions : whether to get reduced attention_mask due to 'down sampling'
        '''

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if mask_time_indices is not None:
            mask_time_indices = mask_time_indices.to(torch.bool)


        all_hidden_states, attention_mask = self.wav2vec2.reduced_forward(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            mask_time_indices=mask_time_indices,
            return_dict=return_dict,
        )

        return all_hidden_states, attention_mask