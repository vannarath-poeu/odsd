import * as React from 'react';
import { Stack, Typography, Box, IconButton, TextField, Button } from '@mui/material';
import {
  MenuRounded,
  LogoutOutlined,
  SendRounded,
} from '@mui/icons-material';

export default function HomePage() {
  const [messages, setMessages] = React.useState<any[]>([]);

  function send() {
    const msgEl:any = document.getElementById("message-input");
    const msg:string = msgEl?.value;
    if (msg) {
      if (msgEl) {
        msgEl.value = "";
      }
      setMessages(messages.concat([{
        message: msg,
        isSpam: msg.indexOf("spam") != -1,
      }]));
      const msgStack = document.getElementById("message-stack");
      if (msgStack) {
        const scrollHeight = msgStack.scrollHeight;
        msgStack.scrollTop = (scrollHeight + 168);
      }
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
          {messages.map(msg => (
            <Stack
              sx={{
                padding: '16px',
                backgroundColor: msg.isSpam ? 'coral' : 'lightblue',
                marginLeft: '24px',
                borderRadius: '8px',
                alignItems: 'flex-start'
              }}
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
