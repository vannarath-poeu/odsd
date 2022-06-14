import * as React from 'react';
import { Stack, Typography, Box, IconButton, TextField, Button } from '@mui/material';
import {
  MenuRounded,
  LogoutOutlined,
  SendRounded,
} from '@mui/icons-material';

import * as tf from '@tensorflow/tfjs';

export default function HomePage() {
  const url: any = {
    model: '/model/model.json',
    metadata: '/model/metadata.json',
  };
  
  const [messages, setMessages] = React.useState<any[]>([]);

  async function loadModel(url: any) {
    try {
      const model: any = await tf.loadGraphModel(url.model);
      setModel(model);
    } 
    catch (err) {
      console.log(err);
    }
  }

  async function loadMetadata(url: any) {
    try {
      const metadataJson = await fetch(url.metadata);
      const metadata = await metadataJson.json();
      setMetadata(metadata);} 
    catch (err) {
      console.log(err);
    }
  }

  const [metadata, setMetadata] = React.useState<any>();
  const [model, setModel] = React.useState<any>();

  React.useEffect(()=>{
    tf.ready().then(()=>{
      loadModel(url);
      loadMetadata(url);
    });
  },[])

  function send() {
    const msgEl:any = document.getElementById("message-input");
    const msg:string = msgEl?.value;
    if (msg) {
      if (msgEl) {
        msgEl.value = "";
      }

      const inputText = msg.trim().toLowerCase().replace(/(\.|\,|\!)/g, '').split(' ');
      const OOV_INDEX = 0;
      const sequence = inputText.map(word => {
        let wordIndex = metadata["word_index"][word] || OOV_INDEX;
        const VOCAB_SIZE = metadata["vocabulary_size"];
        if (wordIndex > VOCAB_SIZE) {
          wordIndex = OOV_INDEX;
        }
        return wordIndex;
      });
      const PAD_INDEX = 0;
      const padSequences = (sequences:any, maxLen:any, padding = 'post', truncating = 'post', value = PAD_INDEX) => {
        return sequences.map((seq: any) => {
          if (seq.length > maxLen) {
            if (truncating === 'pre') {
              seq.splice(0, seq.length - maxLen);
            } else {
              seq.splice(maxLen, seq.length - maxLen);
            }
          }
          if (seq.length < maxLen) {
            const pad = [];
            for (let i = 0; i < maxLen - seq.length; ++i) {
              pad.push(value);
            }
            if (padding === 'pre') {
              seq = pad.concat(seq);
            } else {
              seq = seq.concat(pad);
            }
          }
          return seq;
        });
      }

      const MAX_LENGTH = metadata["max_length"];
      const paddedSequence = padSequences([sequence], MAX_LENGTH);

      const input = tf.tensor2d(paddedSequence, [1, MAX_LENGTH]);

      model.executeAsync(input).then((predictOut: any) => {
        const score: number = predictOut.dataSync()[0];
        predictOut.dispose();

        console.log(score);
  
        setMessages(messages.concat([{
          message: msg,
          isSpam: score >= 0.5,
        }]));
        const msgStack = document.getElementById("message-stack");
        if (msgStack) {
          const scrollHeight = msgStack.scrollHeight;
          msgStack.scrollTop = (scrollHeight + 168);
        }
      })
    }
  }

  return (
    <Stack
      sx={{
        height: '100%',
        width: '100%',
        padding: 0,
      }}
    >
      <Stack
        sx={{
          flexDirection: 'row',
          justifyContent: 'space-between',
          padding: '16px',
        }}
      >
        <IconButton aria-label="menu">
          <MenuRounded />
        </IconButton>
        <IconButton aria-label="logout">
          <LogoutOutlined />
        </IconButton>
      </Stack>
      <Box
        sx={{
          backgroundColor: 'bisque',
          padding: '8px',
        }}
      >
        <Typography>This is a demo app without a server. Input typed below is simulated as incoming messages.</Typography>
      </Box>
      <Box
        sx={{
          height: '100%',
          overflowX: 'clip',
          overflowY: 'clip',
          padding: '16px',
        }}
      >
        <Stack
          id="message-stack"
          sx={{
            height: '100%',
            overflowX: 'clip',
            overflowY: 'auto',
            padding: '16px',
            paddingBottom: '128px',
          }}
          direction="column"
          spacing={2}
        >
          {messages.map((msg, i) => (
            <Stack
              sx={{
                padding: '16px',
                backgroundColor: msg.isSpam ? 'coral' : 'lightblue',
                marginLeft: '24px',
                borderRadius: '8px',
                alignItems: 'flex-start'
              }}
              key={i}
            >
              <div>
                {msg.message}
              </div>
              {msg.isSpam && (
                <Stack
                  sx={{ alignSelf: 'flex-end' }}
                >
                  <Button size="small" onClick={() => alert("Reported")}>Report Spam</Button>
                </Stack>
              )}
            </Stack>
          ))}
        </Stack>
      </Box>
      <Stack
        sx={{
          width: '100%',
          justifyContent: 'space-between',
          padding: '16px',
          // position: 'fixed',
          // bottom: 0,
          height: '96px',
          borderTop: '1px solid #efefef',
          backgroundColor: 'white',
        }}
        direction="row"
        spacing={1}
      >
        <TextField
          id="message-input"
          label='Message'
          variant='outlined'
          required
          sx={{
            width: '100%',
          }}
        />
        <IconButton aria-label="send" onClick={() => send()}>
          <SendRounded />
        </IconButton>
      </Stack>
    </Stack>
  );
}
