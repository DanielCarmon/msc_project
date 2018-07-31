#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This is a prototype of a simple state machine that remotely executes a list of
commands via SSH on network devices (expecting the prompt_pattern to match) and
returns the results to you.
"""

__author__ = 'Jathan McCollum'
__maintainer__ = 'Jathan McCollum'
__email__ = 'jathan@gmail.com'
__version__ = '0.2'

from dcdb import *
import time
import logging
import paramiko
import re
import select
import socket
import sys
import pdb
import os

PROMPT_PATTERN = r'\S+#'

logging.basicConfig(level=logging.DEBUG)
log = logging

#paramiko.util.log_to_file('demo.log')
paramiko.util.log_to_file('/a/home/cc/cs/carmonda/log_tunnel.txt')

class Channel(object):
    def __init__(self, host, commands,log_file,
                 prompt_pattern=PROMPT_PATTERN, init_commands=None):
        self.host = host
        self.commands = commands
        self.username = 'carmonda'
        self.password = ''
        self.log_file = log_file
        self.prompt = re.compile(prompt_pattern)
        if init_commands is None:
            init_commands = []
        self.init_commands = init_commands

        self.results = []

        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        self.ssh = ssh
        self.initialized = False

    def run(self):
        """
        This is what we call to actually connect and start the event loop
        """
        # Establish the SSH connection
        self.ssh.connect(self.host, username=self.username,
                         password=self.password)

        # Create an SSH shell descriptor. This is how we're going to interact
        # with the remote device.
        shell = self.ssh.invoke_shell()
        shell.settimeout(0.0)
        self.shell = shell

        # Turn the list of commands into an iterator, so we know when we've run
        # out of commands
        self.cmditer = iter(self.commands)

        # Establish the data buffer we'll use to store output between commands
        self.data = ''

        # And start the event loop to store the results of the commands and
        # return them when we're done.
        results = self.interact()
        return results

    def interact(self):
        """Interact with the device using the SSH shell."""
        shell = self.shell

        # Start an infinite while loop, and use the select.select async I/O
        # handler to detect when data has been received by the SSH shell.
        while True:
            # The ``r`` variable names the object that has received data. See:
            # http://docs.python.org/2/library/select.html#select.select
            r, w, e = select.select([shell], [], [])
            # If 'shell' has received data, try to retreive it:
            if shell in r:
                #log.debug("HEY LET'S DO SOMETHING WITH SHELL")
                try:
                    # Fetch 1K off the socket.
                    bytes = shell.recv(1024)

                    # If it's no data, we're done. 
                    if len(bytes) == 0:
                        break
                    
                    # Try to process the data we received.
                    self.data_received(bytes)

                # If we timeout or get an error, log it and carry on.
                except (socket.timeout, socket.error) as err:
                    log.error(str(err))

            # If the socket has not received any data after we sent somethign,
            # disconnect.
            else:
                break

        # The explicit call to disconnect
        shell.close()

        # And finally return the output of the results.
        return self.results

    def data_received(self, bytes):
        """
        This is what we do when data is received from the socket.

        :param bytes:
            Bytes that are received.
        """
        # This is our buffer. Until we get a result we want, we keep appending
        # bytes to the data buffer.
        log.debug('Got bytes: %r' % bytes)
        self.data += bytes
        #os.system('echo "{}" >> ~/{} '.format(str(bytes),self.log_file))
        os.system('echo "{}" &>> {} '.format(str(bytes),self.log_file))
        #os.system('sync')
        #log.debug(' Buffered: %r' % self.data)

        # Check if the prompt pattern matches. Return None if it doesn't so the
        # event loop continues to try to receive data. 
        # 
        # Basicaly this means:
        # - Loop until the prompt matches
        # - Trim off the prompt
        # - Store the buffer as the result of the last command sent
        # - Zero out the buffer
        # - Rinse/repeat til all commands are sent and results stored
        m = self.prompt.search(self.data)
        if not m:
            return None
        log.debug('STATE: prompt %r' % m.group())

        # The prompt matched! Strip the prompt from the match result so we get
        # the data received withtout the prompt. This is our result.
        # 
        result = self.data[:m.start()]
        result = result[result.find('\n')+1:]

        # Only keep results once we've started sending commands
        if self.initialized:
            self.results.append(result)
        # And send the next command in the stack.
        self.send_next()

    def send_next(self):
        """
        Send the next command in the command stack.
        """
        
        # We're sending a new command, so we zero out the data buffer.
        self.data = ''

        # Check if we can safely initialize. This is a chance to do setup, such
        # as turning off console paging, or changing up CLI settings. 
        if not self.initialized:
            if self.init_commands:
                next_init = self.init_commands.pop(0)
                self.shell.send(next_init)
                return None
            else:
                log.debug('Successfully initialized for command execution')
                self.initialized = True

        # Try to fetch the next command in the stack. If we're out of commands,
        # close the channel and disconnect.
        try:
            next_command = self.cmditer.next() # Get the next command
        except StopIteration:
            self.close() # Or disconnect
            return None

        # Try to send the next command
        if next_command is None:
            self.results.append(None) # Store a null command w/ null result
            self.send_next() # Fetch the next command
        else:
            log.debug('sending %r' % next_command)
            self.shell.send(next_command + '\n')  # Send this command

    def close(self):
        """Close the SSH connection."""
        self.ssh.close()

if __name__ == '__main__':
    argv = sys.argv
    time.sleep(5)
    username = argv[1]
    host = argv[2]
    cmd = argv[3]
    log_file = argv[4]
    print 'logfile:',log_file,type(log_file)
    prompt_pattern = r'\S+> $'
    init_commands = ['set cli scripting-mode on\n', 'set cli pager off\n']
    commands = [cmd]
    c = Channel(host, commands,log_file ,prompt_pattern, init_commands)
    results = c.run()
    #with open('~/log_main.txt','w') as log_file:
    #    log_file.write(results)
    #print results
